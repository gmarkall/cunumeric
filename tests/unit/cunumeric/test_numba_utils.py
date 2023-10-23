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
import typing
from cunumeric.numba_utils import compile_ptx_soa
from numba.core.typing.templates import Signature
from numba.types import int32, int64, float32, float64, UniTuple


@pytest.fixture
def addsub() -> typing.Callable:
    def addsub(x, y):
        return x + y, x - y

    return addsub


def test_soa(addsub) -> None:
    signature = UniTuple(int32, 2)(int32, int32)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)
    print(ptx)
