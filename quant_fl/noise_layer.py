from abc import abstractmethod
import numbers
import torch
# For Python 2 and 3 support
try:
    from abc import ABC
    from collections.abc import Iterable
except ImportError:
    from abc import ABCMeta
    ABC = ABCMeta('ABC', (), {})
    from collections import Iterable
    
class BasePruningMethod(ABC):
    r"""Abstract base class for creation of new pruning techniques.
    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """
    def __init__(self):
        pass

    def __call__(self, module, inputs):
        r"""Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using
        :meth:`apply_mask`.
        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.
        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """
        pass

    def apply_mask(self, module):
        r"""Simply handles the multiplication between the parameter being
        pruned and the generated mask.
        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.
        Args:
            module (nn.Module): module containing the tensor to prune
        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        assert (
            self._tensor_name is not None
        ), "Module {} has to be pruned".format(
            module
        )  # this gets set in apply()
        # mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        mask = self.compute_mask(orig)
        module._buffers[self._tensor_name + "_mask"].copy_(mask)
        
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, **kwargs):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """
        method = cls(*args, **kwargs)
        method._tensor_name = name
        # at this point we have no forward_pre_hooks but we could have an
        # active reparametrization of the tensor if another pruning method
        # had been applied (in which case `method` would be a PruningContainer
        # and not a simple pruning method).

        # Pruning is to be applied to the module's tensor named `name`,
        # starting from the state it is found in prior to this iteration of
        # pruning
        orig = getattr(module, name)

        # If this is the first time pruning is applied, take care of moving
        # the original tensor to a new parameter called name + '_orig' and
        # and deleting the original parameter

        # copy `module[name]` to `module[name + '_orig']`
        module.register_parameter(name + "_orig", orig)
        # temporarily delete `module[name]`
        del module._parameters[name]



        # Use try/except because if anything goes wrong with the mask
        # computation etc., you'd want to roll back.
        try:
            # get the final mask, computed according to the specific method
            # mask = method.compute_mask(orig)
            # reparametrize by saving mask to `module[name + '_mask']`...
            # module.register_buffer(name + "_mask", mask)
            module.register_buffer(name + "_mask", torch.ones_like(orig))
            # ... and the new pruned tensor to `module[name]`
            setattr(module, name, method.apply_mask(module))
            # associate the pruning method to the module via a hook to
            # compute the function before every forward() (compile by run)
            module.register_forward_pre_hook(method)

        except Exception as e:
            orig = getattr(module, name + "_orig")
            module.register_parameter(name, orig)
            del module._parameters[name + "_orig"]
            raise e

        return method

    def prune(self, t):
        return t * self.compute_mask(t)

    def remove(self, module):
        r"""Removes the pruning reparameterization from a module. The pruned
        parameter named ``name`` remains permanently pruned, and the parameter
        named ``name+'_orig'`` is removed from the parameter list. Similarly,
        the buffer named ``name+'_mask'`` is removed from the buffers.
        Note:
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        assert (
            self._tensor_name is not None
        ), "Module {} has to be pruned\
            before pruning can be removed".format(
            module
        )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        setattr(module, self._tensor_name, orig)


        
        
class RelativeUniform(BasePruningMethod):
    def __init__(self, scale):
        self.scale = scale
    def compute_mask(self, t):
        return torch.empty_like(t).uniform_(-self.scale, self.scale).add_(torch.ones_like(t)).detach()
    
    @classmethod
    def apply(cls, module, name, scale):
        return super(RelativeUniform, cls).apply(module, name, scale=scale)
    
class AbsUniform(BasePruningMethod):
    def __init__(self, scale):
        self.scale = scale
    def compute_mask(self, t):
        return torch.empty_like(t).uniform_(-self.scale, self.scale).div_(t+1e-8).add_(torch.ones_like(t)).detach()
    
    @classmethod
    def apply(cls, module, name, scale):
        return super(AbsUniform, cls).apply(module, name, scale=scale)
    
def relative_uniform(module, name, scale):
    RelativeUniform.apply(module, name, scale)
    return module

def abs_uniform(module, name, scale):
    AbsUniform.apply(module, name, scale)
    return module



##################
## clip weights
##################
