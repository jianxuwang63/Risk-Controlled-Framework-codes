# Third-party model notice

## Phikon

The HistoNexa-MIP Cost-5 checkpoints incorporate a fine-tuned Phikon backbone:

- Creator: Owkin
- Original model: <https://huggingface.co/owkin/phikon>
- Original source repository: <https://github.com/owkin/HistoSSLscaling>
- License: Owkin non-commercial license
- License text: <https://github.com/owkin/HistoSSLscaling/blob/main/LICENSE.txt>

Phikon is a self-supervised Vision Transformer for histopathology. The
HistoNexa-MIP authors modified the pretrained model through downstream
fine-tuning and added selective-classification heads and an ensemble inference
pipeline for the research task documented in this repository. This project is
not sponsored, endorsed, or granted official status by Owkin.

The Owkin license limits the licensed material, derivative works, and results
to non-commercial research purposes by non-profit entities. Each recipient is
responsible for reading and complying with the complete license before using or
redistributing the checkpoints. Commercial and clinical-production use is not
granted. The Apache-2.0 license at the repository root applies only to the
authors' original source code and documentation; it does not relicense Phikon,
any Phikon-derived checkpoint, or any other third-party material.

No Phikon-derived checkpoint is currently distributed from the public source
repository or a GitHub Release while written redistribution authorization is
being confirmed.

## Research and clinical-use boundary

The packaged checkpoints and software are supplied only for academic
reproducibility review. They are not a cleared medical device and must not be
used as the sole basis for diagnosis, prognosis, staging, surgery, or treatment.

The authors retain responsibility for confirming that they have institutional
authority to distribute the fine-tuned checkpoints. This notice does not itself
grant rights beyond those provided by the applicable licenses.
