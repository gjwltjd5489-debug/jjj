# 기술적 지표 100점 매수/매도 점수

일목균형표, 이동평균선, VR(거래량 지표), OBV, RSI, MACD를 합쳐 0~100점 점수와 매수/매도 신호를 만든다.
QQQ 기준으로 시작했고 종목별 프로파일로 확장할 수 있다. **연구용이며 실전 매매용이 아니다.**

- 점수 기준표: [docs/scoring_table.md](docs/scoring_table.md)
- 종목별 파라미터: `scoring/profiles.py`

## 데이터 준비 (Investing.com)

1. Investing.com에서 종목 페이지 → **과거 데이터(Historical Data)** 로 이동
   (QQQ: Invesco QQQ Trust, `investing.com/etfs/powershares-qqqq-historical-data`)
2. 기간 **일간(Daily)**, 시작일을 넉넉히(2년 이상, 가능하면 10년 이상) 설정
3. **다운로드** (로그인 필요) → `data/QQQ.csv` 등으로 저장
   - 한 번에 받을 수 있는 기간이 제한되면 기간을 나눠 여러 파일로 받아도 된다 (자동으로 합치고 중복 제거)
   - 영문 사이트 / 한국어 사이트 CSV 모두 읽을 수 있다

## 실행

```bash
pip install -r requirements.txt

# 최신 날짜 점수표
python scripts/run_score.py --ticker QQQ data/QQQ.csv

# 특정 날짜 점수표
python scripts/run_score.py --ticker QQQ data/QQQ.csv --date 2022-10-13

# 점수 구간별 이후 20/60일 수익률 통계 + VR 분포 + 전체 결과 저장
python scripts/run_score.py --ticker QQQ data/QQQ_part1.csv data/QQQ_part2.csv --eval --out out/qqq.csv
```

출력 예시 (합성 데이터):

```
[2025-09-30] 종가 267.68  총점 62.0 / 100  → 중립
  ichimoku   20 / 30
  ma         26 / 30
  volume      8 / 20
  momentum    8 / 20
```

## 테스트

```bash
python -m pytest -q
```
