# 제출물 원본 (artifacts)

8/31 최종 실행이 끝나면 아래 파일들이 여기에 커밋됩니다.
심사 시 GPU 없이 집계 단계를 그대로 재현하기 위한 자료입니다.

| 파일 | 내용 |
|---|---|
| `submission.csv` | 구글 폼에 제출한 답안 원본 |
| `candidates.csv` | 문제·프롬프트·샘플별 파싱 결과 (생성 텍스트에서 추출한 정수) |
| `test_ids.csv` | 제출 순서를 정의하는 id 목록 |
| `run_report.json` | 실행 시점의 전체 인자와 통계 |

## 집계 재현 (GPU 불필요, 수 초)

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
