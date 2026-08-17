"""
Compatibility patches for third-party dependencies.

Must be imported before any schnetpack module to apply patches.
"""


def _patch_schnetpack_t_co():
    """
    Patch torch.utils.data.dataloader to expose T_co for SchNetPack compatibility.

    SchNetPack 2.1.1 imports T_co from torch.utils.data.dataloader, but PyTorch 2.5+
    renamed it to _T_co. This patch adds T_co = _T_co so schnetpack can import successfully.
    """
    import torch.utils.data.dataloader as dataloader_module

    if "T_co" not in dataloader_module.__dict__ and "_T_co" in dataloader_module.__dict__:
        dataloader_module.T_co = dataloader_module._T_co


# Apply patch on import
_patch_schnetpack_t_co()
