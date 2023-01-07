from typing import Callable, Iterable, Optional, Tuple

from usb.backend import IBackend

class _DescriptorInfo(str): ...

class USBError(IOError): ...
class USBTimeoutError(USBError): ...
class NoBackendError(ValueError): ...

class Endpoint: ...

class Interface: ...

class Configuration: ...

class Device:
    bLength: int
    bDescriptorType: int
    bcdUSB: int
    bDeviceClass: int
    bDeviceSubClass: int
    bDeviceProtocol: int
    bMaxPacketSize0: int
    idVendor: int
    idProduct: int
    bcdDevice: int
    iManufacturer: int
    iProduct: int
    iSerialNumber: int
    bNumConfigurations: int
    address: Optional[int]
    bus: Optional[int]
    port_number: Optional[int]
    port_numbers: Tuple[int, ...]
    speed: Optional[int]

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def configurations(self) -> Tuple['Device']: ...
    # def __init__(self, dev, backend: IBackend) -> None: ...
    @property
    def langids(self) -> Tuple[int]: ...
    @property
    def serial_number(self) -> str: ...
    @property
    def product(self) -> str: ...
    @property
    def parent(self) -> Optional['Device']: ...
    @property
    def manufacturer(self) -> str: ...
    # @property
    # def backend(self) -> IBackend: ...
    # def set_configuration(self, configuration: Optional[int] = None) -> None: ...
    # def get_active_configuration(self) -> Configuration: ...
    # def set_interface_altsetting(self, interface: Optional[int] = None, alternate_setting: Optional[int] = None) -> None: ...
    # def clear_halt(self, ep: Union[int, Endpoint]) -> None: ...
    # def reset(self) -> None: ...
    # def write(self, endpoint: int, data: Union[array.array, str, bytes, None], timeout: Optional[int] = None): ...
    # def read(self, endpoint: int, size_or_buffer: Union[int, array.array], timeout: Optional[int] = None): ...
    # def ctrl_transfer(self, bmRequestType: int, bRequest: int, wValue: int = 0, wIndex: int = 0, data_or_wLength:  Union[array.array, str, bytes, None, int] = None, timeout = None): ...
    # def is_kernel_driver_active(self, interface: int) -> bool: ...
    # def detach_kernel_driver(self, interface: int) -> None: ...
    # def attach_kernel_driver(self, interface: int) -> None: ...
    # def __iter__(self) -> Iterator[Configuration]: ...
    # def __getitem__(self, index: int) -> Configuration: ...
    # @property
    # def default_timeout(self) -> float: ...
    # def finalize(self) -> None: ...

def find(find_all: bool = ..., backend: IBackend = ..., custom_match: Optional[Callable[[Device], bool]] = ...) -> Iterable[Device]: ...
def show_devices(verbose: bool = ...) -> _DescriptorInfo: ...
