import torch

def tensor_product(u, v):
    u_dim = u.dim() - 1
    v_dim = v.dim() - 1
    b = u.shape[0]
    new_u_shape = [u.shape[0]] + [i for i in u.shape[1:]] + [1 for _ in range(v_dim)]
    new_v_shape = [v.shape[0]] + [1 for _ in range(u_dim)] + [i for i in v.shape[1:]]
    return u.view(*new_u_shape) * v.view(*new_v_shape)

class SO3Exp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, w):
        # w: b * 3
        # R: b * 3 * 3
        ctx.save_for_backward(w)
        b = w.shape[0]
        theta = torch.linalg.norm(w, dim=1).view(b, 1, 1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        t1 = torch.eye(3, device=w.device).view(1, 3, 3) * c
        
        # consider 0/0 case
        t2 = tensor_product(w, w) * torch.where(theta == 0., 0.5, (1 - c) / (theta ** 2))
        
        # skew symmetric matrix
        z_skew = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], device=w.device).view(1, 3, 3)
        y_skew = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], device=w.device).view(1, 3, 3)
        x_skew = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], device=w.device).view(1, 3, 3)
        skew_mat = w[:, 0:1, None] * x_skew + w[:, 1:2, None] * y_skew + w[:, 2:3, None] * z_skew
        t3 = skew_mat * torch.where(theta == 0., 1, s / theta)
        
        R = t1 + t2 + t3
        return R
    
    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        b = w.shape[0]
        theta = torch.linalg.norm(w, dim=1).view(b, 1, 1, 1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        x_skew = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], device=w.device).view(1, 3, 3)
        y_skew = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], device=w.device).view(1, 3, 3)
        z_skew = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], device=w.device).view(1, 3, 3)
        e_x = torch.tensor([1, 0, 0], device=w.device).view(1, 3)
        e_y = torch.tensor([0, 1, 0], device=w.device).view(1, 3)
        e_z = torch.tensor([0, 0, 1], device=w.device).view(1, 3)
        
        t1 = tensor_product(x_skew, e_x) + tensor_product(y_skew, e_y) + tensor_product(z_skew, e_z)
        t1 = t1 - tensor_product(torch.eye(3, device=w.device).view(1, 3, 3).expand(b, -1, -1), w)
        t1= t1 * torch.where(theta == 0., 1, s / theta)
        
        t2_x = tensor_product((tensor_product(e_x.expand(b, -1), w) + tensor_product(w, e_x.expand(b, -1))), e_x.expand(b, -1))
        t2_y = tensor_product((tensor_product(e_y.expand(b, -1), w) + tensor_product(w, e_y.expand(b, -1))), e_y.expand(b, -1))
        t2_z = tensor_product((tensor_product(e_z.expand(b, -1), w) + tensor_product(w, e_z.expand(b, -1))), e_z.expand(b, -1))
        t2 = (t2_x + t2_y + t2_z) * torch.where(theta == 0., 0.5, (1 - c) / (theta ** 2))
        
        t3 = tensor_product(tensor_product(w, w), w) * torch.where(theta == 0., -1/12, (2*c -2 + theta*s) / (theta**4))
        
        skew_mat = w[:, 0:1, None] * x_skew + w[:, 1:2, None] * y_skew + w[:, 2:3, None] * z_skew
        t4 = tensor_product(skew_mat, w) * torch.where(theta == 0., -1/3, (theta * c - s) / (theta ** 3))
        
        dRdw = t1 + t2 + t3 + t4 # b*3*3*3
        dw = torch.sum(grad_output.unsqueeze(-1) * dRdw, dim=[1, 2])
        return dw
