import biorbd as biorbd_eigen
import biorbd_casadi as biorbd
import numpy as np
from biorbd import InverseKinematics
from biorbd_casadi import get_range_q
from casadi import MX, vertcat, sumsqr, nlpsol, horzcat, SX, Function

import constants


class MarkerOptimization:
    def __init__(self, model, marker_data: np.ndarray, with_root: bool = False, marker_bounds: dict[dict] = None):
        """
        Parameters
        ----------
        model: biorbd.Model
            The biorbd model loaded with biorbd CasADi
        marker_data: np.ndarray
            The position of the markers from the c3d of shape (nb_dim, nb_marker, nb_frame),
            nb_marker should be equal to the number of markers in the model, unit should be in meters.
        """
        self.with_root = with_root

        self.biorbd_model = model
        self.biorbd_model_eigen = biorbd_eigen.Model(
            self.biorbd_model.path().absolutePath().to_string()
        )
        self.nb_q = self.biorbd_model.nbQ()
        self.marker_names = [
            self.biorbd_model.markerNames()[i].to_string()
            for i in range(len(self.biorbd_model.markerNames()))
        ]
        self.nb_markers_model = self.biorbd_model.nbMarkers()
        self.marker_bounds = marker_bounds
        self.nb_markers_to_optimize = len(self.marker_bounds)
        self.marker_names_to_optimize = list(self.marker_bounds.keys())
        self.marker_names_static = list(set(self.marker_names) - set(self.marker_names_to_optimize))
        self.marker_idx_to_optimize = [self.marker_names.index(marker) for marker in self.marker_names_to_optimize]

        if isinstance(marker_data, np.ndarray):
            if (
                    marker_data.ndim >= 2
                    and marker_data.shape[0] <= 3
                    and marker_data.shape[1] == self.nb_markers_model
            ):
                self.xp_markers = marker_data
                self.nb_frames = marker_data.shape[2]
            else:
                raise ValueError(
                    f"The standard dimension of the NumPy array should be (nb_dim, nb_marker, nb_frame)"
                )
        else:
            raise ValueError("The standard inputs is a numpy.ndarray")

        if with_root:
            self.root_mx = MX.sym("root", 3, 1)
            # symbolic in the model now
            rt = biorbd.RotoTrans.fromEulerAngles(MX([0, 0, 0]), self.root_mx, "xyz")
            self.biorbd_model.segments()[1].setLocalJCS(self.biorbd_model, rt)

        self.marker_mx, self.vert_marker_mx = self._declare_marker_sym()
        self.q_mx, self.vert_q_mx = self._declare_q_sym()
        # self.x_mx = vertcat(self.vert_q_mx, self.vert_marker_mx)
        if with_root:
            self.x_mx = vertcat(self.vert_q_mx, self.vert_marker_mx, self.root_mx)
        else:
            self.x_mx = vertcat(self.vert_q_mx, self.vert_marker_mx)

        self.use_sx = True

        self.q_bounds = get_range_q(self.biorbd_model)

        self.indices_to_remove = []
        self.indices_to_keep = []
        self._get_nan_index()

        self.list_sol = []

        self.output = dict()
        self.nb_dim = self.xp_markers.shape[0]

    def _declare_q_sym(self) -> tuple[MX, MX]:
        """Declares the symbolic variables for the natural coordinates and handle single frame or multi frames"""
        q_sym = []
        for f in range(self.nb_frames):
            q_f_sym = MX.sym(f"q_{f}", self.nb_q, 1)
            q_sym.append(q_f_sym)
        q = horzcat(*q_sym)
        vert_q = vertcat(*q_sym)
        return q, vert_q

    def _declare_marker_sym(self) -> tuple[MX, MX]:
        """Declares the symbolic variables for the natural coordinates and handle single frame or multi frames"""
        marker_sym_list = []
        for i in range(self.nb_markers_model):
            print(self.marker_names[i])
            if self.marker_names[i] in self.marker_names_to_optimize:
                marker_i_sym = MX.sym(f"m_{i}", 3, 1)
                marker_sym_list.append(marker_i_sym)

        marker_sym = horzcat(*marker_sym_list)
        vert_marker_sym = vertcat(*marker_sym_list)
        return marker_sym, vert_marker_sym

    def _get_nan_index(self):
        """
        Find, for each frame, the index of the markers which has a nan value
        """
        for j in range(self.nb_frames):
            self.indices_to_remove.append(
                list(np.unique(np.isnan(self.xp_markers[:, :, j]).nonzero()[1]))
            )
            self.indices_to_keep.append(
                list(np.unique(np.isfinite(self.xp_markers[:, :, j]).nonzero()[1]))
            )

    def _objective_minimize_marker_distance(
            self, q, experimental_markers, marker_location_mx
    ) -> MX:
        """
        Computes the objective function that minimizes marker distance and handles single frame or multi frames

        Returns
        -------
        MX
            The objective function that minimizes the distance between the experimental markers and the model markers
        """
        error_m = 0
        for f in range(self.nb_frames):
            qf = q[:, f]
            marker_location = compute_marker_location_mx(
                self.biorbd_model, qf, marker_location_mx
            )
            xp_markers = (
                experimental_markers[:3, :, f]
                if isinstance(experimental_markers, np.ndarray)
                else experimental_markers
            )
            # error_m += sumsqr(marker_location - xp_markers[:, self.indices_to_keep[f]])
            error_m += sumsqr(marker_location - xp_markers.flatten("F"))

        return error_m

    def solve(
            self,
            method: str = "ipopt",
            options: dict = None,
            marker_bounds: dict[dict] = None,
    ):
        """
        Solve the optimization problem to find the generalized coordinates and marker parameters
        which minimize the difference between the markers' position in the model and in the c3d

        Parameters:
        ----------

        Returns
        ----------
        marker_parameters: np.ndarray [3, nb_marker]
        """

        default_options = {
            "sqpmethod": constants.SQP_IK_VALUES,
            "ipopt": constants.IPOPT_IK_VALUES,
        }

        options = options or default_options.get(method)
        if options is None:
            raise ValueError(
                "method must be one of the following str: 'sqpmethod' or 'ipopt'"
            )

        initial_root = self.biorbd_model_eigen.localJCS(0).to_array()[:3, 3]

        initial_marker_location = np.array(
            [
                self.biorbd_model_eigen.marker(i, False).to_array()
                for i in self.marker_idx_to_optimize
            ]
        ).T
        vert_initial_marker_location = initial_marker_location.flatten("F")

        # Ik to get a good initial guess
        ik = InverseKinematics(self.biorbd_model_eigen, self.xp_markers)
        q_init = ik.solve()
        vert_q_init = q_init.flatten("F")

        if self.with_root:
            x0_init = np.concatenate(
                (vert_q_init, vert_initial_marker_location, initial_root), axis=0
            )
        else:
            x0_init = np.concatenate(
                (vert_q_init, vert_initial_marker_location), axis=0
            )

        bounds = self.q_bounds
        lbx = np.repeat(bounds[0][:, np.newaxis], self.nb_frames, axis=1).flatten("F")
        ubx = np.repeat(bounds[1][:, np.newaxis], self.nb_frames, axis=1).flatten("F")

        # add marker_bounds
        if self.marker_bounds is None:
            lbx = np.concatenate(
                (lbx, np.repeat(-np.inf, self.nb_dim * self.nb_markers_model))
            )
            ubx = np.concatenate(
                (ubx, np.repeat(np.inf, self.nb_dim * self.nb_markers_model))
            )
        else:
            mlbx = []
            mubx = []
            for _, marker_bound in self.marker_bounds.items():
                mlbx = np.concatenate((mlbx, np.array(marker_bound["lbx"])), axis=0)
                mubx = np.concatenate((mubx, np.array(marker_bound["ubx"])), axis=0)

            lbx = np.concatenate((lbx, mlbx), axis=0)
            ubx = np.concatenate((ubx, mubx), axis=0)

        # add root_bounds
        if self.with_root:
            lbx = np.concatenate((lbx, np.repeat(-np.inf, 3)))
            ubx = np.concatenate((ubx, np.repeat(np.inf, 3)))

        marker_locations_mx = self.marker_location_vector()

        objective = self._objective_minimize_marker_distance(
            self.q_mx, self.xp_markers, marker_locations_mx
        )

        nlp = dict(
            x=self.x_mx,
            f=_mx_to_sx(objective, [self.x_mx]) if self.use_sx else objective,
        )

        r, success = _solve_nlp(method, nlp, x0_init, lbx=lbx, ubx=ubx, options=options)

        x_opt = r["x"].toarray()
        q_opt = x_opt[: self.nb_q * self.nb_frames, :].reshape(
            self.nb_q, self.nb_frames, order="F"
        )
        marker_parameters = x_opt[
                            (self.nb_q * self.nb_frames): (
                                    self.nb_q * self.nb_frames + self.nb_dim * self.nb_markers_to_optimize
                            ),
                            :,
                            ].reshape(self.nb_dim, self.nb_markers_to_optimize, order="F")
        root = x_opt[-3:, :] if self.with_root else None

        return q_opt, marker_parameters, root, success

    def marker_location_vector(self):
        """ Returns the marker location as a vector
        with symbolic mx and numeric mx for the marker that should not move """
        marker_location = []
        j = 0
        for i in range(self.nb_markers_model):
            if self.marker_names[i] in self.marker_names_to_optimize:
                marker_location.append(self.marker_mx[:, j])
                j += 1
            else:
                marker_location.append(self.biorbd_model.marker(i, False).to_mx())

        marker_location_mx = horzcat(*marker_location)
        return marker_location_mx

    def print_result(self, result):
        print("marker_location :", result[1])
        for i in range(result[1].shape[1]):
            idx_marker_to_optimize = self.marker_idx_to_optimize[i]
            print("marker_name :", self.biorbd_model_eigen.markerNames()[idx_marker_to_optimize].to_string())
            print("now marker_location :", result[1][:, i])
            print("before marker_location :", self.biorbd_model_eigen.marker(idx_marker_to_optimize, False).to_array())
            print("\n")

        if self.with_root:
            print("now root_location :", result[2][0:3, 0])
            print("before root_location :", self.biorbd_model_eigen.localJCS(1).trans().to_array())

        for i in range(result[1].shape[1]):
            idx_marker_to_optimize = self.marker_idx_to_optimize[i]
            print(f"marker\t{self.biorbd_model_eigen.markerNames()[idx_marker_to_optimize].to_string()}")
            segment_id = self.biorbd_model_eigen.marker(idx_marker_to_optimize, False).parentId()
            print(f"\tparent\t{self.biorbd_model_eigen.segments()[segment_id].name().to_string()}")
            print(f"\tposition\t{result[1][0, i]}\t{result[1][1, i]}\t{result[1][2, i]}")
            print(f"endmarker")
            print(f"\n")


def compute_marker_location_mx(
        model: biorbd.Model, q: MX, marker_location_mx: MX
) -> MX:
    """
    Computes the location of the markers in the model

    Parameters
    ----------
    model: biorbd.Model
        The biorbd model loaded with biorbd CasADi
    q: MX
        The generalized coordinates
    marker_location_mx: MX
        The marker location

    Returns
    -------
    MX
        The location of the markers in the model
    """
    marker_location = []
    for i in range(model.nbMarkers()):
        segment_id = model.marker(i, False).parentId()
        T0i = model.globalJCS(q, segment_id).to_mx()
        marker_location.append((T0i @ vertcat(marker_location_mx[:, i], 1))[:3, 0])

    return vertcat(*marker_location)


def _mx_to_sx(mx: MX, symbolics: list[MX]) -> SX:
    """
    Converts a MX to a SX

    Parameters
    ----------
    mx : MX
        The MX to convert
    symbolics : list[MX]
        The symbolics to use

    Returns
    -------
    The converted SX
    """
    f = Function("f", symbolics, [mx]).expand()
    return f(*symbolics)


def _solve_nlp(
        method: str,
        nlp: dict,
        q_init: np.ndarray,
        lbx: np.ndarray = None,
        ubx: np.ndarray = None,
        options: dict = None,
):
    """
    Solves a nonlinear program with CasADi

    Parameters
    ----------
    method : str
        The method to use to solve the NLP (ipopt, sqpmethod, ...)
    nlp : dict
        The NLP to solve
    q_init : np.ndarray
        The initial guess
    lbx : np.ndarray
        The lower q_bounds
    ubx : np.ndarray
        The upper q_bounds
    options : dict
        The options to pass to the solver

    Returns
    -------
    The output of the solver
    """
    S = nlpsol("MarkerOptimization", method, nlp, options)
    r = S(x0=q_init, lbx=lbx, ubx=ubx)

    if S.stats()["success"] is False:
        print("MarkerOptimization failed to converge")

    return r, S.stats()["success"]
