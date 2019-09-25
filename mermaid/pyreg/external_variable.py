use_mermaid_net = False
reg_factor_in_mermaid = 1.
update_sm_by_advect = False
update_sm_with_interpolation =False
bysingle_int = False
debug_mode_on = False
try:
    from model_pool.global_variable import use_mermaid_iter, reg_factor_in_mermaid, update_sm_by_advect,update_sm_with_interpolation,bysingle_int
    use_mermaid_net = not use_mermaid_iter
    update_sm_by_advect = update_sm_by_advect
    update_sm_with_interpolation = update_sm_with_interpolation
    bysingle_int = bysingle_int
except:
    pass
print("use_mermaid_net:{}".format(use_mermaid_net))
print("reg_factor_in_mermaid:{}".format(reg_factor_in_mermaid))
use_odeint = True
grads={}
def save_grad(name):
    def hook(grad):
        grads[name]=grad
    return hook
