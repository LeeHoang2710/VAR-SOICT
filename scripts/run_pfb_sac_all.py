#!/usr/bin/env python3
"""Run the complete resumable original-PFB+SAC generation and metric workflow."""

import run_original_pfb_sac_batch as batch


_validate_inputs = batch.runtime.validate_inputs


def _validate_paper_inputs(args):
    args.inject_step = 2
    return _validate_inputs(args)


batch.runtime.validate_inputs = _validate_paper_inputs
batch.main()
