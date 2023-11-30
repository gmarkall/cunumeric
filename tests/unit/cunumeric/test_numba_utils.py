# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
import re
import typing
from cunumeric.numba_utils import compile_ptx_soa
from numba.types import int32, int64, UniTuple


@pytest.fixture
def addsub() -> typing.Callable:
    def addsub(x, y):
        return x + y, x - y

    return addsub


def test_soa(addsub) -> None:
    # A basic test of compilation with an SoA interface

    # Compile for two int32 inputs and two int32 outputs
    signature = UniTuple(int32, 2)(int32, int32)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)

    # The function definition should use the name of the Python function
    fn_def_pattern = r"\.visible\s+\.func\s+addsub"
    assert re.search(fn_def_pattern, ptx)

    # The return type should match that of the signature's return type
    assert resty == signature.return_type

    # The function should have 4 parameters (numbered 0 to 3)
    assert re.search("addsub_param_3", ptx)
    assert not re.search("addsub_param_4", ptx)

    # The first two parameters should be treated as pointers (u64 values)
    assert re.search(r"ld\.param\.u64\s+%rd[0-9]+,\s+\[addsub_param_0\]", ptx)
    assert re.search(r"ld\.param\.u64\s+%rd[0-9]+,\s+\[addsub_param_1\]", ptx)

    # The remaining two parameters should be treated as 32 bit integers
    assert re.search(r"ld\.param\.u32\s+%r[0-9]+,\s+\[addsub_param_2\]", ptx)
    assert re.search(r"ld\.param\.u32\s+%r[0-9]+,\s+\[addsub_param_3\]", ptx)


def test_soa_fn_name(addsub) -> None:
    # Ensure that when a wrapper function name is specified, it is used in the
    # PTX.
    signature = UniTuple(int32, 2)(int32, int32)
    abi_info = {'abi_name': 'addsub_soa'}
    ptx, resty = compile_ptx_soa(addsub, signature, device=True,
                                 abi_info=abi_info)
    fn_def_pattern = r"\.visible\s+\.func\s+addsub_soa"
    assert re.search(fn_def_pattern, ptx)


def test_soa_arg_types(addsub) -> None:
    # Ensure that specifying a different argument type is reflected
    # appropriately in the generated PTX
    signature = UniTuple(int32, 2)(int32, int64)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)

    # The final two parameters should now be a 32- and a 64-bit values
    # respectively. Note that the load of the last parameter may be an
    # instruction with a 32-bit destination type that effectively chops off the
    # upper 32 bits, so we cannot test for a load of a 64-bit value, which
    # would look like:
    #
    #    ld.param.u64 	%rd2, [addsub_param_3];
    #
    # but instead we'd potentially get
    #
    #    ld.param.u32 	%r2, [addsub_param_3];
    #
    # So we test the bit width of the parameters only:
    assert re.search(r".param\s+.b32\s+addsub_param_2", ptx)
    assert re.search(r".param\s+.b64\s+addsub_param_3", ptx)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
