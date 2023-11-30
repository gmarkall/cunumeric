from llvmlite import ir
from numba import types
from numba.core import cgutils, sigutils
from numba.core.callconv import BaseCallConv
from numba.core.compiler_lock import global_compiler_lock
from numba.cuda.compiler import compile_cuda


class SoACallConv(BaseCallConv):
    """
    Calling convention where returned values are stored through pointers
    provided as arguments.

    - If the return type is a scalar, the first argument is a pointer to the
      return type.
    - If the return type is a tuple of length N, then the first N arguments are
      pointers to each of the elements of the tuple.

    In equivalent C, the prototype of a function with this calling convention
    would take the following form:

        void <func_name>(<Tuple item 1>*, ..., <Tuple item N>*,
                         <Python arguments... >);
    """

    def _make_call_helper(self, builder):
        # Not needed for wrapping functions only.
        return None

    def return_value(self, builder, retval):
        return builder.ret(retval)

    def return_user_exc(self, builder, exc, exc_args=None, loc=None,
                        func_name=None):
        msg = "Python exceptions are unsupported when returning in SoA form"
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder, status):
        msg = "Return status is unsupported when returning in SoA form"
        raise NotImplementedError(msg)
        pass

    def get_function_type(self, restype, argtypes):
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        if isinstance(restype, types.BaseTuple):
            return_types = [self.get_return_type(t) for t in restype.types]
        else:
            return_types = [self.get_return_type(restype)]
        fnty = ir.FunctionType(ir.VoidType(), return_types + argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn),
                             ['arg.' + a for a in args])

    def get_arguments(self, func, restype):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        if isinstance(restype, types.BaseTuple):
            n_returns = len(restype.types)
        else:
            n_returns = 1

        return func.args[n_returns:]

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        raise NotImplementedError("Can't call SoA return function directly")


def soa_wrap_function(context, lib, fndesc, nvvm_options,
                      wrapper_name):
    """
    Wrap a Numba ABI function such that it returns tuple values into SoA
    arguments.
    """
    library = lib.codegen.create_library(f'{lib.name}_function_',
                                         entry_name=wrapper_name,
                                         nvvm_options=nvvm_options)
    library.add_linking_library(lib)

    # Determine the caller (C ABI) and wrapper (Numba ABI) function types
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    soa_call_conv = SoACallConv(context)
    wrapperty = soa_call_conv.get_function_type(restype, argtypes)
    calleety = context.call_conv.get_function_type(restype, argtypes)

    # Create a new module and declare the callee
    wrapper_module = context.create_module("cuda.soa.wrapper")
    callee = ir.Function(wrapper_module, calleety, fndesc.llvm_func_name)

    # Define the caller - populate it with a call to the callee and return
    # its return value

    wrapper = ir.Function(wrapper_module, wrapperty, wrapper_name)
    builder = ir.IRBuilder(wrapper.append_basic_block(''))

    arginfo = context.get_arg_packer(argtypes)
    wrapper_args = soa_call_conv.get_arguments(wrapper, restype)
    callargs = arginfo.as_arguments(builder, wrapper_args)
    # We get (status, return_value), but we ignore the status since we
    # can't propagate it through the SoA ABI anyway
    _, return_value = context.call_conv.call_function(
        builder, callee, restype, argtypes, callargs)

    if isinstance(restype, types.BaseTuple):
        for i in range(len(restype.types)):
            val = builder.extract_value(return_value, i)
            builder.store(val, wrapper.args[i])
    else:
        builder.store(return_value, wrapper.args[0])
    builder.ret_void()

    library.add_ir_module(wrapper_module)
    library.finalize()
    return library


@global_compiler_lock
def compile_ptx_soa(pyfunc, sig, debug=False, lineinfo=False, device=False,
                    fastmath=False, cc=None, opt=True, abi_info=None):
    # This is just a copy of Numba's compile_ptx, with a modification to return
    # values as SoA and some simplifications to keep it short
    nvvm_options = {
        'fastmath': fastmath,
        'opt': 3 if opt else 0
    }

    if abi_info:
        wrapper_name = abi_info['abi_name']
    else:
        wrapper_name = pyfunc.__name__

    args, return_type = sigutils.normalize_signature(sig)

    cc = cc or (5, 0)
    cres = compile_cuda(pyfunc, return_type, args, debug=debug,
                        lineinfo=lineinfo, fastmath=fastmath,
                        nvvm_options=nvvm_options, cc=cc)
    resty = cres.signature.return_type

    tgt = cres.target_context

    if device:
        lib = soa_wrap_function(tgt, cres.library, cres.fndesc, nvvm_options,
                                wrapper_name)
    else:
        code = pyfunc.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno

        lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug,
                                              lineinfo, nvvm_options, filename,
                                              linenum)

    ptx = lib.get_asm_str(cc=cc)
    return ptx, resty
