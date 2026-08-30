# 제출물 원본 (artifacts)

8/31 최종 실행이 만든 파일들입니다. 심사 시 **GPU 없이 집계 단계를 그대로 재현**하기
위한 자료입니다.

| 파일 | 내용 |
|---|---|
| `test_submission.csv` | 구글 폼에 제출한 파일. 주최 측 배포 파일의 `answer` 열을 채운 것 (`id, question, answer`) |
| `submission.csv` | 집계 결과 (`id, answer`). 아래 재현의 대조 대상 |
| `candidates.csv` | 문제·프롬프트·샘플별 파싱 결과 — 생성 텍스트에서 추출한 정수 |
| `test_ids.csv` | 제출 순서를 정의하는 id 목록 |
| `run_report.json` | 실행 시점의 전체 인자와 통계 |

## 1. 집계 재현 (GPU 불필요, 수 초)

```bash
python src/aggregate_only.py \
    --candidates artifacts/candidates.csv \
    --test       artifacts/test_ids.csv \
    --out        /tmp/rebuilt.csv
diff /tmp/rebuilt.csv artifacts/submission.csv
```

차이가 없으면, 제출 답안이 `candidates.csv` 의 파싱값만으로 결정되었음이 확인됩니다.
`candidates.csv` 의 각 행은 모델이 생성한 텍스트에서 뽑아낸 정수 하나이며,
정답 키나 외부 데이터가 개입하지 않습니다.

## 2. 제출 파일과의 대조

```bash
python - <<'PY'
import pandas as pd
sub = pd.read_csv("artifacts/submission.csv")
fin = pd.read_csv("artifacts/test_submission.csv")
sub["id"] = sub["id"].astype(str); fin["id"] = fin["id"].astype(str)
m = fin.merge(sub, on="id", suffixes=("_final", "_agg"))
print("행 수      ", len(fin), len(sub), len(m))
print("답 불일치  ", int((m["answer_final"] != m["answer_agg"]).sum()))
print("정수 아닌 행", int((~fin["answer"].astype(str).str.fullmatch(r"-?\d+")).sum()))
PY
```

`답 불일치 0`, `정수 아닌 행 0` 이면 폼에 제출한 파일이 위 재현 결과와 동일합니다.

자세한 절차는 저장소 README 6절에 있습니다.
