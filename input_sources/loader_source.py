from __future__ import annotations

from .frame_source import BaseFrameSource, FrameSourceInfo


class LoaderFrameSource(BaseFrameSource):
    def __init__(self, loader, *, kind: str, sequence_name: str):
        super().__init__(
            FrameSourceInfo(
                kind=kind,
                sequence_name=sequence_name,
                camera=getattr(loader, "camera", "unknown"),
                device_name=getattr(loader, "device_name", "unknown"),
                is_live=False,
            ),
            resize=getattr(loader, "resize", None),
        )
        self.loader = loader

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self) -> int:
        return len(self.loader)

    def set_resize(self, resize):
        super().set_resize(resize)
        self.loader.resize = resize

    def reset(self) -> None:
        if hasattr(self.loader, "index"):
            self.loader.index = 0
        if hasattr(self.loader, "_init_prefetch"):
            self.loader._init_prefetch()

    def __next__(self):
        return next(self.loader)

    def close(self) -> None:
        return
