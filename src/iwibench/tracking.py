"""Idempotent W&B publication of an already measured terminal result."""
from pathlib import Path
import json
import os
import secrets


def log_result(result, project, receipt_path):
    import wandb
    receipt_path = Path(receipt_path)
    identity = {"project":project,"patch_sha256":result["patch"]["sha256"],
                "trace_id":result.get("trace_id",""),"run_type":result.get("run_type","unclassified")}
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get("identity") != identity:
            raise ValueError("W&B receipt identity does not match this publication")
        return previous
    patch = result["patch"]
    metrics = result["metrics"]
    run = wandb.init(project=project,id=Path(patch["path"]).parent.name[:24],resume="allow",
                     config={"model_id":patch["model_id"],"revision":patch["revision"],
                             "trace_id":result.get("trace_id",""),"run_type":result.get("run_type","unclassified"),
                             "family":metrics["family"],"episode_id":metrics["episode_id"]},reinit="finish_previous")
    try:
        values = {k:v for k,v in metrics.items() if isinstance(v,(float,int))}
        values.update(result["counters"])
        values.update({"patch/"+k:patch[k] for k in ["changed_parameters","l2","linf"]})
        values["patch/bytes"] = Path(patch["path"]).stat().st_size
        run.log(values)
        artifact = wandb.Artifact("patch-"+Path(patch["path"]).parent.name,type="weight-patch",metadata=patch)
        artifact.add_file(patch["path"])
        run.log_artifact(artifact)
        receipt = {"run_id":run.id,"url":run.url,"trace_id":result.get("trace_id",""),"identity":identity}
        run.finish()
    except Exception:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass  # Preserve the original upload failure, which prevented logging.
        raise
    temporary = receipt_path.with_name(receipt_path.name+"."+secrets.token_hex(6)+".tmp")
    try:
        temporary.write_text(json.dumps(receipt,indent=2)+"\n")
        os.replace(temporary,receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt
