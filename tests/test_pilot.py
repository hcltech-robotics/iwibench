"""Subprocess orchestration fixtures; these are not editor or model executions."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest

spec = importlib.util.spec_from_file_location("run_pilot",Path(__file__).parents[1]/"scripts/run_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Subprocess fixture did not reach the expected state")


@pytest.fixture
def child_program(tmp_path):
    """A real process that writes labelled fixture traces and launch counters."""
    script = tmp_path/"fixture_child.py"
    script.write_text('''import fcntl,json,sys,time
from pathlib import Path
path,counter,gate = map(Path,sys.argv[1:4])
scored = sys.argv[4] == "scored"
path.mkdir(parents=True,exist_ok=True)
def count(delta):
    with counter.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(counter.read_text()) if counter.exists() else {"active":0,"maximum":0,"launches":0}
        state["active"]+=delta
        state["launches"]+=delta == 1
        state["maximum"]=max(state["maximum"],state["active"])
        temporary=counter.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(counter)
count(1)
(path/"fixture_started").write_text("unit subprocess only")
while not gate.exists(): time.sleep(.01)
trace={"ok":scored,"is_completed":True,"errors":[],"rewards":{"editing_reward":{"score":0,"weight":1}} if scored else {},"info":{"iwibench":{"patch":{"sha256":"a"*64},"metrics":{"reward":0}}}}
(path/"traces").mkdir(exist_ok=True)
(path/"traces"/"traces.jsonl").write_text(json.dumps({"ok":scored,"traces":[trace]})+"\\n")
count(-1)
''')
    return script


def child_command(script, path, counter, gate, scored=True):
    return [sys.executable,str(script),str(path),str(counter),str(gate),"scored" if scored else "failed"]


def test_completion_requires_a_scored_trace(tmp_path):
    assert not pilot.completed_score(tmp_path)
    trace = {"ok":True,"is_completed":True,"rewards":{"editing_reward":{"score":0,"weight":1}},"info":{"iwibench":{"patch":{"sha256":"a"*64},"metrics":{"reward":0}}},"errors":[]}
    path = tmp_path/"traces.jsonl"
    path.write_text(json.dumps({"ok":True,"traces":[trace]})+"\n")
    assert pilot.completed_score(tmp_path)
    trace["ok"] = False
    trace["is_completed"] = True
    path.write_text(json.dumps({"ok":False,"traces":[trace]})+"\n")
    assert not pilot.completed_score(tmp_path)
    trace["ok"] = True
    trace["rewards"] = {}
    path.write_text(json.dumps({"ok":True,"traces":[trace]})+"\n")
    assert not pilot.completed_score(tmp_path)


@pytest.mark.parametrize("completion", [False, None])
def test_valid_reward_without_completed_trace_is_rejected(tmp_path, completion):
    trace = {"ok":True,"rewards":{"editing_reward":{"score":.5,"weight":1}},
             "info":{"iwibench":{"patch":{"sha256":"b"*64},"metrics":{"reward":.5}}},"errors":[]}
    if completion is not None:
        trace["is_completed"] = completion
    path = tmp_path/"traces.jsonl"
    path.write_text(json.dumps({"ok":True,"traces":[trace]})+"\n")
    assert not pilot.completed_score(tmp_path)
    trace["is_completed"] = True
    path.write_text(json.dumps({"ok":True,"traces":[trace]})+"\n")
    assert pilot.completed_score(tmp_path)


def test_parallel_attempts_bound_running_children(tmp_path, child_program):
    counter,gate = tmp_path/"counter.json",tmp_path/"release"
    attempts = [(tmp_path/f"episode-{i}","alias_acquisition",i,
                 child_command(child_program,tmp_path/f"episode-{i}",counter,gate)) for i in range(5)]
    with ThreadPoolExecutor(max_workers=1) as runner:
        pending = runner.submit(pilot.execute_attempts,attempts,2)
        try:
            wait_for(lambda:counter.exists() and json.loads(counter.read_text())["active"] == 2
                     and sum((attempt[0]/"fixture_started").exists() for attempt in attempts) == 2)
            assert sum((attempt[0]/"fixture_started").exists() for attempt in attempts) == 2
            assert not pending.done()
        finally:
            gate.touch()
        assert pending.result(timeout=10) == 0
    counts = json.loads(counter.read_text())
    assert counts == {"active":0,"maximum":2,"launches":5}
    assert all(pilot.completed_score(attempt[0]/"traces") for attempt in attempts)


def test_existing_live_pid_is_waited_and_recovered(tmp_path, child_program):
    path,counter,gate = tmp_path/"episode",tmp_path/"counter.json",tmp_path/"release"
    command = child_command(child_program,path,counter,gate)
    process = subprocess.Popen(command)
    try:
        wait_for(lambda:(path/"fixture_started").exists())
        pilot.write_status(path/"execution.json",{"pid":process.pid,"family":"alias_acquisition","seed":1,"returncode":None})
        with ThreadPoolExecutor(max_workers=1) as runner:
            pending = runner.submit(pilot.run_attempt,path,"alias_acquisition",1,command,.01)
            try:
                time.sleep(.05)
                assert process.poll() is None
                assert not pending.done()
            finally:
                gate.touch()
            assert pending.result(timeout=10)
    finally:
        gate.touch()
        process.wait(timeout=10)
    assert json.loads(counter.read_text())["launches"] == 1
    status = json.loads((path/"execution.json").read_text())
    assert status["returncode"] is None
    assert status["completed_score"] is True and status["recovered"] is True


def test_supervisor_exit_keeps_child_claim_and_resume_does_not_duplicate(tmp_path, child_program):
    path,counter,gate = tmp_path/"episode",tmp_path/"counter.json",tmp_path/"release"
    command = child_command(child_program,path,counter,gate)
    source = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('pilot',sys.argv[1]); "
        "pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot); "
        "sys.exit(0 if pilot.run_attempt(Path(sys.argv[2]),'alias_acquisition',1,json.loads(sys.argv[3]),.01) else 1)"
    )
    launch = [sys.executable,"-c",source,str(Path(pilot.__file__)),str(path),json.dumps(command)]
    first = subprocess.Popen(launch)
    second = None
    try:
        wait_for(lambda:(path/"fixture_started").exists())
        first.terminate()
        first.wait(timeout=10)
        second = subprocess.Popen(launch)
        time.sleep(.1)
        assert second.poll() is None
        assert json.loads(counter.read_text())["launches"] == 1
        gate.touch()
        assert second.wait(timeout=10) == 0
    finally:
        gate.touch()
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=10)
        if second is not None and second.poll() is None:
            second.terminate()
            second.wait(timeout=10)
    status = json.loads((path/"execution.json").read_text())
    assert status["returncode"] is None and status["recovered"] is True
    assert json.loads(counter.read_text())["launches"] == 1


def test_concurrent_claims_launch_one_attempt(tmp_path, child_program):
    path,counter,gate = tmp_path/"episode",tmp_path/"counter.json",tmp_path/"release"
    command = child_command(child_program,path,counter,gate)
    with ThreadPoolExecutor(max_workers=2) as runner:
        pending = [runner.submit(pilot.run_attempt,path,"alias_acquisition",1,command,.01) for _ in range(2)]
        try:
            wait_for(lambda:(path/"fixture_started").exists())
        finally:
            gate.touch()
        assert all(future.result(timeout=10) for future in pending)
    assert json.loads(counter.read_text())["launches"] == 1
    status = json.loads((path/"execution.json").read_text())
    assert status["returncode"] == 0 and status["completed_score"] is True
    assert "recovered" not in status


def test_zero_exit_without_score_fails_and_cannot_relaunch(tmp_path, child_program):
    path,counter,gate = tmp_path/"episode",tmp_path/"counter.json",tmp_path/"release"
    gate.touch()
    command = child_command(child_program,path,counter,gate,scored=False)
    assert pilot.execute_attempts([(path,"alias_acquisition",1,command)],1) == 1
    status = json.loads((path/"execution.json").read_text())
    assert status["returncode"] == 0 and status["completed_score"] is False
    assert not pilot.run_attempt(path,"alias_acquisition",1,command)
    assert json.loads(counter.read_text())["launches"] == 1


def test_completed_trace_without_supervisor_status_is_recovered(tmp_path, child_program):
    path,counter,gate = tmp_path/"episode",tmp_path/"counter.json",tmp_path/"release"
    gate.touch()
    command = child_command(child_program,path,counter,gate)
    subprocess.run(command,check=True)
    assert not (path/"execution.json").exists()
    assert pilot.run_attempt(path,"alias_acquisition",1,command)
    status = json.loads((path/"execution.json").read_text())
    assert status["returncode"] is None
    assert status["completed_score"] is True and status["recovered"] is True
    assert json.loads(counter.read_text())["launches"] == 1


def test_partial_trace_and_invalid_workers_are_rejected(tmp_path):
    (tmp_path/"traces.jsonl").write_text('{"ok":')
    assert not pilot.completed_score(tmp_path)
    with pytest.raises(ValueError,match="workers"):
        pilot.execute_attempts([],0)


@pytest.mark.parametrize("effort", [None, "low"])
def test_pilot_forwards_explicit_reasoning_effort(tmp_path, monkeypatch, effort):
    from iwibench.families import FAMILY_VERSION

    calibration = tmp_path/"calibration.json"
    calibration.write_text(json.dumps({
        "family_version":FAMILY_VERSION,"model_id":"google/gemma-3-4b-it",
        "episodes":[{"eligible_for_edit_attempt":True,
                     "episode":{"family":"alias_acquisition","parameter_seed":1}}],
    }))
    arguments = ["run_pilot.py",str(calibration),"--service-env",str(tmp_path/"credentials"),
                 "--output-dir",str(tmp_path/"runs"),"--editor-model","gpt-6.1-sol","--workers","4"]
    if effort is not None:
        arguments.extend(["--reasoning-effort",effort])
    captured = {}

    def capture(attempts, workers):
        captured.update(attempts=attempts,workers=workers)
        return 0

    monkeypatch.setattr(sys,"argv",arguments)
    monkeypatch.setattr(pilot,"execute_attempts",capture)
    assert pilot.main() == 0
    assert captured["workers"] == 4
    assert len(captured["attempts"]) == 1
    command = captured["attempts"][0][3]
    assert command[command.index("--editor-model")+1] == "gpt-6.1-sol"
    if effort is None:
        assert "--reasoning-effort" not in command
    else:
        assert command[command.index("--reasoning-effort")+1] == effort
