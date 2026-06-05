import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# settings
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_DIM = 2          # [theta, theta_dot]
CONTROL_DIM = 1
LATENT_DIM = 8
HIDDEN_DIM = 64

DT = 0.05
ROLLOUT_STEPS = 30     # 1.5 seconds
TRAIN_TRAJS = 1500
TEST_TRAJS = 300

TRAIN_EPOCHS = 120
BATCH_SIZE = 64
LR = 1e-3


# utilities
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# damped pendulum environment
# x = [theta, theta_dot]
class DampedPendulumEnv:
    def __init__(self, dt=DT, g=9.81, l=1.0, m=1.0, b=0.15, u_max=2.0):
        self.dt = dt
        self.g = g
        self.l = l
        self.m = m
        self.b = b
        self.u_max = u_max

    def reset(self, batch_size=1, near_upright=False):
        if near_upright:
            theta = np.random.uniform(-0.3, 0.3, size=(batch_size, 1))
            theta_dot = np.random.uniform(-0.3, 0.3, size=(batch_size, 1))
        else:
            theta = np.random.uniform(-math.pi, math.pi, size=(batch_size, 1))
            theta_dot = np.random.uniform(-1.0, 1.0, size=(batch_size, 1))

        x = np.concatenate([theta, theta_dot], axis=1)
        return x.astype(np.float32)

    def step(self, x, u):
        theta = x[:, 0:1]
        theta_dot = x[:, 1:2]
        u = np.clip(u, -self.u_max, self.u_max)

        # nonlinear state-dependent control term
        theta_ddot = (
            -(self.g / self.l) * np.sin(theta)
            - self.b * theta_dot
            + np.cos(theta) * u / (self.m * self.l)
        )

        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_ddot

        x_next = np.concatenate([theta_next, theta_dot_next], axis=1)
        x_next[:, 0] = angle_normalize(x_next[:, 0])
        return x_next.astype(np.float32)

    def rollout(self, x0, U):
        x = x0.copy()
        X = []

        for t in range(U.shape[1]):
            u = U[:, t, :]
            x = self.step(x, u)
            X.append(x.copy())

        return np.stack(X, axis=1)

    def cost(self, x, u):
        theta = x[:, 0]
        theta_dot = x[:, 1]
        u = u[:, 0]
        return theta**2 + 0.1 * theta_dot**2 + 0.01 * u**2


# data collection
def collect_dataset(env, num_trajs, horizon, near_upright=False):
    x0_all = []
    U_all = []
    X_all = []

    for _ in range(num_trajs):
        x0 = env.reset(batch_size=1, near_upright=near_upright)
        U = np.random.uniform(-env.u_max, env.u_max, size=(1, horizon, 1)).astype(np.float32)
        X = env.rollout(x0, U)

        x0_all.append(x0[0])
        U_all.append(U[0])
        X_all.append(X[0])

    x0_all = np.array(x0_all, dtype=np.float32)
    U_all = np.array(U_all, dtype=np.float32)
    X_all = np.array(X_all, dtype=np.float32)

    return x0_all, U_all, X_all


# local linear baseline
# linearized near upright theta = 0
class LocalLinearBaseline:
    def __init__(self, dt=DT, g=9.81, l=1.0, m=1.0, b=0.15):
        self.A = np.array([
            [1.0, dt],
            [-(g / l) * dt, 1.0 - b * dt]
        ], dtype=np.float32)

        self.B = np.array([
            [0.0],
            [dt / (m * l)]
        ], dtype=np.float32)

    def rollout(self, x0, U):
        batch_size = x0.shape[0]
        horizon = U.shape[1]
        X_pred = np.zeros((batch_size, horizon, STATE_DIM), dtype=np.float32)

        x = x0.copy()
        for t in range(horizon):
            u = U[:, t, :]  # [B,1]
            x_next = np.zeros_like(x)

            for i in range(batch_size):
                xi = x[i].reshape(-1, 1)
                ui = u[i].reshape(1, 1)
                xn = self.A @ xi + self.B @ ui
                xn = xn.flatten()
                xn[0] = angle_normalize(xn[0])
                x_next[i] = xn

            x = x_next
            X_pred[:, t, :] = x

        return X_pred


# EDMD baseline
# z_{k+1} = A z_k + B u_k
def edmd_phi(x):
    theta = x[..., 0]
    theta_dot = x[..., 1]

    obs = np.stack([
        theta,
        theta_dot,
        np.sin(theta),
        np.cos(theta),
        theta * theta_dot
    ], axis=-1)

    return obs.astype(np.float32)


class EDMDBaseline:
    def __init__(self):
        self.A = None
        self.B = None

    def fit(self, x0_train, U_train, X_train):
        Z_list = []
        Znext_list = []
        U_list = []

        for i in range(x0_train.shape[0]):
            x = x0_train[i]
            for t in range(U_train.shape[1]):
                z = edmd_phi(x)
                z_next = edmd_phi(X_train[i, t])
                u = U_train[i, t]

                Z_list.append(z)
                Znext_list.append(z_next)
                U_list.append(u)

                x = X_train[i, t]

        Z = np.array(Z_list)          # [N, dz]
        Znext = np.array(Znext_list)  # [N, dz]
        U = np.array(U_list)          # [N, du]

        Xreg = np.concatenate([Z, U], axis=1)  # [N, dz+du]

        # least squares: Znext = Xreg @ W
        W, _, _, _ = np.linalg.lstsq(Xreg, Znext, rcond=None)
        dz = Z.shape[1]

        self.A = W[:dz, :].T
        self.B = W[dz:, :].T

    def z_to_state(self, z):
        # first two observables are theta, theta_dot
        x = z[..., :2].copy()
        x[..., 0] = angle_normalize(x[..., 0])
        return x

    def rollout(self, x0, U):
        batch_size = x0.shape[0]
        horizon = U.shape[1]
        X_pred = np.zeros((batch_size, horizon, STATE_DIM), dtype=np.float32)

        z = edmd_phi(x0)  # [B,dz]

        for t in range(horizon):
            u = U[:, t, :]  # [B,1]
            z = (z @ self.A.T) + (u @ self.B.T)
            x = self.z_to_state(z)
            X_pred[:, t, :] = x

        return X_pred


# deep koopman model
class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(STATE_DIM, LATENT_DIM)

    def forward(self, x):
        extra = self.net(x)
        z = torch.cat([x, extra], dim=1)
        return z


class DKUCControl(nn.Module):
    def forward(self, x, u):
        return u


class DKACControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(STATE_DIM, CONTROL_DIM)

    def forward(self, x, u):
        scale = self.net(x)
        return scale * u

    def invert(self, x, u_hat):
        scale = self.net(x)
        scale = torch.where(torch.abs(scale) < 1e-3,
                            torch.ones_like(scale) * 1e-3,
                            scale)
        return u_hat / scale


class DKNControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(STATE_DIM + CONTROL_DIM, CONTROL_DIM)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        return self.net(xu)


class DeepKoopman(nn.Module):
    def __init__(self, model_type="DKAC"):
        super().__init__()

        self.model_type = model_type
        self.encoder = Encoder()
        self.lifted_dim = STATE_DIM + LATENT_DIM

        if model_type == "DKUC":
            self.control_net = DKUCControl()
        elif model_type == "DKAC":
            self.control_net = DKACControl()
        elif model_type == "DKN":
            self.control_net = DKNControl()
        else:
            raise ValueError("model_type must be DKUC, DKAC, or DKN")

        self.A = nn.Linear(self.lifted_dim, self.lifted_dim, bias=False)
        self.B = nn.Linear(CONTROL_DIM, self.lifted_dim, bias=False)

        self.reset_linear_weights()

    def reset_linear_weights(self):
        with torch.no_grad():
            eye = torch.eye(self.lifted_dim)
            self.A.weight.copy_(eye + 0.01 * torch.randn_like(eye))
            self.B.weight.copy_(0.01 * torch.randn_like(self.B.weight))

    def lift(self, x):
        return self.encoder(x)

    def get_state_from_z(self, z):
        x = z[:, :STATE_DIM].clone()
        x[:, 0] = ((x[:, 0] + np.pi) % (2 * np.pi)) - np.pi
        return x

    def step(self, x, u):
        z = self.lift(x)
        u_hat = self.control_net(x, u)
        z_next = self.A(z) + self.B(u_hat)
        x_next = self.get_state_from_z(z_next)
        return x_next, z_next

    def rollout(self, x0, U):
        x = x0
        out = []

        for t in range(U.shape[1]):
            u = U[:, t, :]
            x, _ = self.step(x, u)
            out.append(x)

        return torch.stack(out, dim=1)

    def recover_control(self, x, u_hat):
        if self.model_type == "DKUC":
            return u_hat
        elif self.model_type == "DKAC":
            return self.control_net.invert(x, u_hat)
        else:
            raise NotImplementedError("DKN control recovery not implemented.")


# deep koopman training
def multi_step_loss(model, x0, U, X_true, gamma=0.95):
    X_pred = model.rollout(x0, U)

    loss = 0.0
    for t in range(X_true.shape[1]):
        step_loss = F.mse_loss(X_pred[:, t, :], X_true[:, t, :])
        loss = loss + (gamma ** t) * step_loss

    return loss


def make_batches(x0, U, X, batch_size=BATCH_SIZE):
    n = x0.shape[0]
    idx = np.random.permutation(n)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield x0[batch_idx], U[batch_idx], X[batch_idx]


def train_deep_model(model, x0_train, U_train, X_train, epochs=TRAIN_EPOCHS):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x0_train_t = torch.tensor(x0_train, dtype=torch.float32)
    U_train_t = torch.tensor(U_train, dtype=torch.float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for bx0, bU, bX in make_batches(x0_train_t, U_train_t, X_train_t):
            bx0 = bx0.to(DEVICE)
            bU = bU.to(DEVICE)
            bX = bX.to(DEVICE)

            optimizer.zero_grad()
            loss = multi_step_loss(model, bx0, bU, bX)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{model.model_type} epoch {epoch+1:3d} | loss = {avg_loss:.6f}")

    return losses


# evaluation metrics
def prediction_metrics(X_pred, X_true):
    mean_mse = np.mean((X_pred - X_true) ** 2)
    final_mse = np.mean((X_pred[:, -1, :] - X_true[:, -1, :]) ** 2)
    step_mse = np.mean((X_pred - X_true) ** 2, axis=(0, 2))
    return mean_mse, final_mse, step_mse


@torch.no_grad()
def evaluate_deep_model(model, x0, U, X_true):
    model.eval()

    x0_t = torch.tensor(x0, dtype=torch.float32, device=DEVICE)
    U_t = torch.tensor(U, dtype=torch.float32, device=DEVICE)

    X_pred = model.rollout(x0_t, U_t).cpu().numpy()
    return prediction_metrics(X_pred, X_true), X_pred


# LQR helpers
def solve_dare_iterative(A, B, Q, R, max_iter=500, tol=1e-8):
    P = Q.copy()
    for _ in range(max_iter):
        BtPB = B.T @ P @ B
        inv_term = np.linalg.inv(R + BtPB)
        P_next = A.T @ P @ A - A.T @ P @ B @ inv_term @ B.T @ P @ A + Q
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = P_next
    return P


def compute_lqr_gain(A, B, Q, R):
    P = solve_dare_iterative(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


# local linear control
def run_local_linear_control(env, baseline, x_init, steps=100):
    Q = np.diag([10.0, 1.0]).astype(np.float32)
    R = np.array([[0.1]], dtype=np.float32)
    K = compute_lqr_gain(baseline.A, baseline.B, Q, R)

    x = x_init.reshape(1, -1).astype(np.float32)
    traj = [x[0].copy()]
    costs = []

    for _ in range(steps):
        u = -(K @ x.T).T
        u = np.clip(u, -env.u_max, env.u_max).astype(np.float32)

        costs.append(env.cost(x, u)[0])
        x = env.step(x, u)
        traj.append(x[0].copy())

    return np.array(traj), np.array(costs)


# deep koopman control
@torch.no_grad()
def run_deep_control(env, model, x_init, steps=100):
    if model.model_type == "DKN":
        raise ValueError("DKN control not used because inverse control recovery is not implemented.")

    model.eval()

    A = model.A.weight.detach().cpu().numpy()
    B = model.B.weight.detach().cpu().numpy()

    Q = np.eye(model.lifted_dim, dtype=np.float32)
    Q[0, 0] = 10.0
    Q[1, 1] = 1.0
    R = np.array([[0.1]], dtype=np.float32)

    K = compute_lqr_gain(A, B, Q, R)

    x = x_init.reshape(1, -1).astype(np.float32)
    traj = [x[0].copy()]
    costs = []

    for _ in range(steps):
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        z = model.lift(x_t).cpu().numpy()

        u_hat = -(K @ z.T).T.astype(np.float32)
        u_hat_t = torch.tensor(u_hat, dtype=torch.float32, device=DEVICE)

        u_real = model.recover_control(x_t, u_hat_t).cpu().numpy()
        u_real = np.clip(u_real, -env.u_max, env.u_max).astype(np.float32)

        costs.append(env.cost(x, u_real)[0])
        x = env.step(x, u_real)
        traj.append(x[0].copy())

    return np.array(traj), np.array(costs)


# plotting
def plot_prediction_errors(results):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    names = list(results.keys())
    mean_vals = [results[k]["mean_mse"] for k in names]
    plt.bar(names, mean_vals)
    plt.ylabel("Mean rollout MSE")
    plt.title("Prediction error over 30-step horizon")

    plt.subplot(1, 2, 2)
    for name in names:
        plt.plot(results[name]["step_mse"], label=name)
    plt.xlabel("Prediction step")
    plt.ylabel("Step MSE")
    plt.title("Prediction error by step")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_example_rollout(X_true, rollout_dict, sample_idx=0):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(X_true[sample_idx, :, 0], label="ground truth", linewidth=2)
    for name, X_pred in rollout_dict.items():
        plt.plot(X_pred[sample_idx, :, 0], label=name)
    plt.xlabel("Step")
    plt.ylabel("theta")
    plt.title("Example theta rollout")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(X_true[sample_idx, :, 1], label="ground truth", linewidth=2)
    for name, X_pred in rollout_dict.items():
        plt.plot(X_pred[sample_idx, :, 1], label=name)
    plt.xlabel("Step")
    plt.ylabel("theta_dot")
    plt.title("Example theta_dot rollout")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_control_results(control_results):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for name, item in control_results.items():
        traj = item["traj"]
        plt.plot(traj[:, 0], traj[:, 1], label=name)
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.title("Control phase portrait")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, item in control_results.items():
        costs = item["costs"]
        plt.plot(np.cumsum(costs), label=name)
    plt.xlabel("Step")
    plt.ylabel("Cumulative cost")
    plt.title("Control cost")
    plt.legend()

    plt.tight_layout()
    plt.show()


# main experiment
def main():
    print("Using device:", DEVICE)

    env = DampedPendulumEnv()

    print("\nCollecting training and test data...")
    x0_train, U_train, X_train = collect_dataset(env, TRAIN_TRAJS, ROLLOUT_STEPS, near_upright=False)
    x0_test, U_test, X_test = collect_dataset(env, TEST_TRAJS, ROLLOUT_STEPS, near_upright=False)

    print("\nRunning local linear baseline...")
    local_linear = LocalLinearBaseline()
    X_pred_lin = local_linear.rollout(x0_test, U_test)
    mean_mse_lin, final_mse_lin, step_mse_lin = prediction_metrics(X_pred_lin, X_test)

    print("Running EDMD baseline...")
    edmd = EDMDBaseline()
    edmd.fit(x0_train, U_train, X_train)
    X_pred_edmd = edmd.rollout(x0_test, U_test)
    mean_mse_edmd, final_mse_edmd, step_mse_edmd = prediction_metrics(X_pred_edmd, X_test)

    print("\nTraining deep Koopman models...")
    dkuc = DeepKoopman("DKUC")
    dkac = DeepKoopman("DKAC")
    dkn = DeepKoopman("DKN")

    loss_dkuc = train_deep_model(dkuc, x0_train, U_train, X_train)
    loss_dkac = train_deep_model(dkac, x0_train, U_train, X_train)
    loss_dkn = train_deep_model(dkn, x0_train, U_train, X_train)

    (m_dkuc, f_dkuc, s_dkuc), X_pred_dkuc = evaluate_deep_model(dkuc, x0_test, U_test, X_test)
    (m_dkac, f_dkac, s_dkac), X_pred_dkac = evaluate_deep_model(dkac, x0_test, U_test, X_test)
    (m_dkn, f_dkn, s_dkn), X_pred_dkn = evaluate_deep_model(dkn, x0_test, U_test, X_test)

    results = {
        "LocalLinear": {
            "mean_mse": mean_mse_lin,
            "final_mse": final_mse_lin,
            "step_mse": step_mse_lin,
        },
        "EDMD": {
            "mean_mse": mean_mse_edmd,
            "final_mse": final_mse_edmd,
            "step_mse": step_mse_edmd,
        },
        "DKUC": {
            "mean_mse": m_dkuc,
            "final_mse": f_dkuc,
            "step_mse": s_dkuc,
        },
        "DKAC": {
            "mean_mse": m_dkac,
            "final_mse": f_dkac,
            "step_mse": s_dkac,
        },
        "DKN": {
            "mean_mse": m_dkn,
            "final_mse": f_dkn,
            "step_mse": s_dkn,
        },
    }

    print("\nPrediction results:")
    for name, res in results.items():
        print(
            f"{name:12s} | mean rollout MSE = {res['mean_mse']:.6f} | "
            f"final-step MSE = {res['final_mse']:.6f}"
        )

    plot_prediction_errors(results)

    rollout_dict = {
        "LocalLinear": X_pred_lin,
        "EDMD": X_pred_edmd,
        "DKUC": X_pred_dkuc,
        "DKAC": X_pred_dkac,
        "DKN": X_pred_dkn,
    }
    plot_example_rollout(X_test, rollout_dict, sample_idx=0)

    print("\nRunning control experiment...")
    x_init = np.array([0.25, 0.15], dtype=np.float32)

    control_results = {}

    traj_lin, costs_lin = run_local_linear_control(env, local_linear, x_init, steps=80)
    control_results["LocalLinear"] = {"traj": traj_lin, "costs": costs_lin}

    traj_dkuc, costs_dkuc = run_deep_control(env, dkuc, x_init, steps=80)
    control_results["DKUC"] = {"traj": traj_dkuc, "costs": costs_dkuc}

    traj_dkac, costs_dkac = run_deep_control(env, dkac, x_init, steps=80)
    control_results["DKAC"] = {"traj": traj_dkac, "costs": costs_dkac}

    print("\nControl results:")
    for name, item in control_results.items():
        final_state = item["traj"][-1]
        total_cost = np.sum(item["costs"])
        print(
            f"{name:12s} | total cost = {total_cost:.6f} | "
            f"final state = [{final_state[0]:.4f}, {final_state[1]:.4f}]"
        )

    plot_control_results(control_results)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_dkuc, label="DKUC")
    plt.plot(loss_dkac, label="DKAC")
    plt.plot(loss_dkn, label="DKN")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Deep Koopman training loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
