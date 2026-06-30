"""Smoke test: retrieval grid with 2-cell mini-grid runs end-to-end."""
import pytest
import yaml

from eval.runs.retrieval_grid import run


@pytest.mark.integration
def test_retrieval_grid_smoke(tmp_path):
    grid = {
        "top_k": [5],
        "variant": ["default"],
        "hybrid": [False, True],
        "rerank": [False],
        "query_rewrite": [False],
    }
    grid_file = tmp_path / "test_grid.yaml"
    grid_file.write_text(yaml.dump(grid))

    out_path = run(grid_yaml=grid_file)

    assert out_path.exists()
    content = out_path.read_text()
    lines = content.strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "mean_ndcg_at_k" in lines[0]
