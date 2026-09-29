# 기술적 지표 100점 매수/매도 점수

일목균형표, 이동평균, 거래량(VR·분산일), 모멘텀, 변동성, 추세강도, 상대강도, 매크로 지표로 0~100점 점수와 매수/매도 신호를 만든다.
QQQ 기준으로 시작했고, 종목별 프로파일로 확장할 수 있다. **연구용이며 실전 매매용이 아니다.**

## 문서
| 문서 | 내용 |
|---|---|
| [docs/scoring_table.md](docs/scoring_table.md) | v0 초기 기준표 |
| [docs/review.md](docs/review.md) | v0 비판적 검토 (실제 나스닥 데이터 기반), 추가 카테고리 추천 |
| [docs/scorecards.md](docs/scorecards.md) | 새 기준표 v1 균형형 · v2 추세추종형 · v3 눌림목형 · v4 리스크관리형 + 비교 결과 |
| [docs/rotation.md](docs/rotation.md) | v5 조합 카드(v3 진입 + v2 청산), 저상관 바스켓 선정, 상위 n 로테이션 전략과 검증 |
| [docs/data_sources.md](docs/data_sources.md) | 주가 데이터를 바로 받을 수 있는 사이트 검토 |
| [docs/checklist_review.md](docs/checklist_review.md) | 일일 체크리스트 적정성 검토와 v2 (성향별 판정) |

## 설치
```bash
pip install -r requirements.txt
pip install finance-datareader     # 선택: 인터넷에서 바로 받기 (fdr:)
pip install arch --no-deps         # 선택: 내장 샘플 데이터 (sample:nasdaq/sp500/vix/wti)
pip install zipline-reloaded --no-deps  # 선택: 국채금리 샘플 (sample:longbond/cash)
```

## 데이터 지정 방법
| 형식 | 예 | 설명 |
|---|---|---|
| CSV 경로 | `data/QQQ.csv` | Investing.com 과거 데이터 CSV (영문/한국어), Yahoo·FDR 형식. 여러 개 넣으면 합침 |
| `fdr:심볼` | `fdr:QQQ`, `fdr:INVESTING:QQQ`, `fdr:005930`, `fdr:FRED:DGS10` | FinanceDataReader 로 바로 받기 |
| `fdr:A+B` | `fdr:PDBC+DBC` | A 상장 전 구간을 B 수익률로 이어 붙임 |
| `sample:이름` | `sample:nasdaq`, `sample:sp500`, `sample:vix`, `sample:wti`, `sample:longbond`, `sample:cash` | 패키지 내장 실제 데이터 (인터넷 불필요) |

Investing.com CSV 받기: 종목 페이지 → **과거 데이터(Historical Data)** → 일간, 시작일 넉넉히(2년 이상) → **다운로드**(로그인 필요).

## 실행
```bash
# 최신 날짜 점수표 (카드 하나)
python scripts/run_score.py data/QQQ.csv --card v2

# 모든 카드 한 번에 + 상대강도 벤치마크 + 매크로
python scripts/run_score.py fdr:QQQ --card all --bench fdr:SPY \
    --vix fdr:FRED:VIXCLS --us10y fdr:FRED:DGS10 --hy fdr:FRED:BAMLH0A0HYM2

# 특정 날짜 점수표 + 구간별 이후 수익률·전략 성과
python scripts/run_score.py sample:nasdaq --bench sample:sp500 --card v3 --date 2008-10-10 --eval

# 카드 비교 (전체 + 기간 분할)
python scripts/compare_cards.py sample:nasdaq --bench sample:sp500 --split 2009-01-01 --md out/compare.md

# 바스켓 로테이션: QQQ VWO GLD PDBC EDV UUP 중 v5 점수 상위 2개 보유
python scripts/rotate_basket.py --top 2 --cash fdr:BIL
python scripts/rotate_basket.py --demo     # 인터넷 없이 대용 데이터로
```

## 구조
- `scoring/indicators.py` — 지표 계산 (일목, 이평, VR, OBV, RSI, MACD, ADX, 볼린저 %B, 실현변동성, 분산일 등)
- `scoring/cards.py` — 카드(기준표) 정의: 항목별 조건·배점·신호 임계값
- `scoring/profiles.py` — 종목별 지표 파라미터 (이평 기간, VR 구간 등)
- `scoring/score.py` — 점수 계산 엔진 (선택 카테고리 환산, 게이트, 평활, 이벤트)
- `scoring/evaluate.py` — 구간별 이후 수익률, 순위상관(IC), 이벤트 전략 성과
- `scoring/basket.py` — 후보 유니버스, 수익률·점수 상관, 저상관 종목 선정
- `scoring/portfolio.py` — 상위 n 로테이션 백테스트, 동일가중 비교
- `scoring/sources.py`, `scoring/data.py` — CSV / FinanceDataReader / 샘플 데이터 로더

## 새 카드 만들기
`scoring/cards.py`에서 `Item(key, category, max, fn, desc)` 목록으로 `Card`를 만들고 `CARDS`에 등록한다.
배점 합계가 100이 아니면 import 시점에 오류가 난다.

## 테스트
```bash
python -m pytest -q
```
미래 정보 누수 테스트가 포함되어 있다. 데이터 앞부분만으로 계산한 점수가 전체 데이터로 계산한 같은 날 점수와 같은지 검사한다.
