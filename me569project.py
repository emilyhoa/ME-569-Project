import torch
import torch.nn as nn
import torch.nn.functional as F


# just putting settings here for now
STATE_DIM = 4
CONTROL_DIM = 1
LATENT_DIM = 8
HIDDEN_DIM = 64
ROLLOUT_STEPS = 10
MODEL_TYPE = "DKAC"   # DKUC, DKAC, DKN


# simple neural net
class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# encoder for z = [x ; extra features]
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(STATE_DIM, LATENT_DIM)

    def forward(self, x):
        extra = self.net(x)
        z = torch.cat([x, extra], dim=1)
        return z


# control versions
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


class DKNControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(STATE_DIM + CONTROL_DIM, CONTROL_DIM)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        return self.net(xu)


# main model
class DeepKoopman(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        lifted_dim = STATE_DIM + LATENT_DIM

        if MODEL_TYPE == "DKUC":
            self.control_net = DKUCControl()
        elif MODEL_TYPE == "DKAC":
            self.control_net = DKACControl()
        else:
            self.control_net = DKNControl()

        self.A = nn.Linear(lifted_dim, lifted_dim, bias=False)
        self.B = nn.Linear(CONTROL_DIM, lifted_dim, bias=False)

    def lift(self, x):
        return self.encoder(x)

    def get_state_from_z(self, z):
        return z[:, :STATE_DIM]

    def step(self, x, u):
        z = self.lift(x)
        u_hat = self.control_net(x, u)
        z_next = self.A(z) + self.B(u_hat)
        x_next = self.get_state_from_z(z_next)
        return x_next

    def rollout(self, x0, U):
        x = x0
        out = []

        for t in range(U.shape[1]):
            u = U[:, t, :]
            x = self.step(x, u)
            out.append(x)

        out = torch.stack(out, dim=1)
        return out


def multi_step_loss(model, x0, U, X_true):
    X_pred = model.rollout(x0, U)

    loss = 0
    for t in range(X_true.shape[1]):
        step_loss = F.mse_loss(X_pred[:, t, :], X_true[:, t, :])
        loss = loss + (0.95 ** t) * step_loss

    return loss


# fake batch for now just so code runs
def make_fake_batch(batch_size):
    x0 = torch.randn(batch_size, STATE_DIM)
    U = torch.randn(batch_size, ROLLOUT_STEPS, CONTROL_DIM) * 0.1
    X = torch.randn(batch_size, ROLLOUT_STEPS, STATE_DIM)

    # need real trajectories here later
    return x0, U, X


def train_one_step(model, opt, x0, U, X):
    model.train()
    opt.zero_grad()
    loss = multi_step_loss(model, x0, U, X)
    loss.backward()
    opt.step()
    return loss.item()


# not done yet
def collect_data_from_env():
    # need to connect this to actual environment
    pass


def test_model():
    # need real eval, maybe rollout error by step
    pass


def run_control():
    # need LQR stuff here later
    pass


def compare_models():
    # need to run DKUC vs DKAC vs DKN
    pass


def make_plots():
    # should probably save some graphs for the report
    pass


# main
model = DeepKoopman()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    x0, U, X = make_fake_batch(32)
    loss = train_one_step(model, opt, x0, U, X)
    print("epoch", epoch + 1, "loss =", loss)

print("done for now")