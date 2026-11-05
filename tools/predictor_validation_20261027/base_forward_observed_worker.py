"""Separate semantic observer for PEFT generation that bypasses outer forward.

The original observer is immutable for its running pilots. PEFT's generate
delegates to its base causal LM, whose forward is observed here without any
additional synchronization or inference changes. This is a new context.
"""
import semantic_observed_worker as semantic


def install_base_forward(adapter):
    if adapter.executor.family!='chatqa':
        return
    model=adapter.model
    if not getattr(model,'peft_config',None) or not callable(getattr(model,'get_base_model',None)):
        raise ValueError('Explicit PEFT ChatQA generation owner required')
    owner=model.get_base_model()
    if owner is model or not callable(getattr(owner,'forward',None)) or not callable(getattr(owner,'generate',None)):
        raise ValueError('Distinct callable PEFT base causal LM required')
    semantic.observe_method(owner,'forward','model_forward')


def main():
    original=semantic.install_compact
    def install(adapter):
        original(adapter)
        install_base_forward(adapter)
    semantic.install_compact=install
    semantic.main()


if __name__=='__main__':
    main()
