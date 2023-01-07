# Copyright (c) 2021 Chris Reed
#
# SPDX-License-Identifier: Apache-2.0
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

import functools
from typing import Any, Callable, Iterable, Optional, Union
import usb

from usb.backend import IBackend
from pathlib import Path


@functools.lru_cache()
def get_library_path() -> Optional["Path"]: ...


def find_library(candidate: str) -> Optional[str]: ...


@functools.lru_cache()
def get_libusb1_backend() -> Optional["IBackend"]: ...


def find(find_all: bool, custom_match: Optional[Callable[[usb.core.Device], bool]] = None) -> Iterable[usb.core.Device]: ...
