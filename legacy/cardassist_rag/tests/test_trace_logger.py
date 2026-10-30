import json
from src import trace_logger

def test_in_place_node_retains_before_and_after(tmp_path, monkeypatch):
    path = tmp_path / 'trace.jsonl'
    monkeypatch.setattr(trace_logger, '_logger', trace_logger.TraceLogger(str(path)))
    @trace_logger.traced('mutating_node')
    def node(state):
        state['items'].append('new')
        state['answer'] = 'done'
        return state
    node({'run_id':'test', 'items':['old']})
    row = json.loads(path.read_text())
    assert row['inputs']['items'] == ['old']
    assert row['outputs'] == {'items':['old','new'], 'answer':'done'}
