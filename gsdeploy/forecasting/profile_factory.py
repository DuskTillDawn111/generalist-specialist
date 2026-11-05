"""Explicit choice of one context model for the shared FIFO engine."""
from .profiles import ProfileProvider


def execution_profile_provider(document,execution_context_sha256,*,allow_cpu_fixture=False):
    if 'native_activity_law' in document:
        from .native_context import NativeContextProvider
        return NativeContextProvider(document,execution_context_sha256,allow_cpu_fixture=allow_cpu_fixture)
    return ProfileProvider(document,execution_context_sha256,allow_cpu_fixture=allow_cpu_fixture)
