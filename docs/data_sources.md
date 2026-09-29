# 주가 데이터를 바로 받을 수 있는 곳 (검토, 2026-09 기준)

## 결론
| 용도 | 1순위 | 대안 |
|---|---|---|
| 미국 ETF·주식 일봉 (QQQ, SPY) | **FinanceDataReader** `fdr:QQQ` (Yahoo 경유, 키 불필요) | yfinance, Tiingo(무료 키, 공식 API) |
| Investing.com 데이터를 그대로 | 웹에서 **CSV 다운로드** (가장 확실) | FinanceDataReader `fdr:INVESTING:QQQ` (비공식) |
| 금리·스프레드·VIX | **FRED** `fdr:FRED:DGS10`, `fdr:FRED:BAMLH0A0HYM2`, `fdr:FRED:VIXCLS` | CBOE VIX_History.csv |
| 한국 종목·지수 | **FinanceDataReader** `fdr:005930` (네이버 경유) | 한국투자증권 KIS Open API(공식), pykrx(KRX 로그인 필요) |

이 저장소의 스크립트는 모두 지원한다:
```bash
pip install finance-datareader
python scripts/run_score.py fdr:QQQ --card all --bench fdr:SPY \
    --vix fdr:FRED:VIXCLS --us10y fdr:FRED:DGS10 --hy fdr:FRED:BAMLH0A0HYM2
python scripts/run_score.py fdr:INVESTING:QQQ --card v2          # Investing.com 소스
python scripts/compare_cards.py fdr:QQQ --bench fdr:SPY --split 2016-01-01
```

> 이 검토는 개발 환경(클라우드 세션)의 네트워크 정책 때문에 사이트 **실시간 접속 확인은 못 했다**.
> - 각 라이브러리 설치 파일의 소스코드(FinanceDataReader 0.9.202, pykrx 1.2.9, yfinance 1.7.0)에서 실제 호출 주소와 인증 방식을 확인했다.
> - 그 밖의 내용은 알려진 정책을 정리한 것이다.
> - 무료 한도와 정책은 자주 바뀌므로 쓰기 전에 각 사이트 문서를 확인할 것.

## 소스별 정리

### 1. Investing.com
- **공식 API는 없다.**
- **웹 CSV 다운로드**: 종목 페이지 → Historical Data → Download. 로그인이 필요하다. 가장 안정적이며 이 저장소의 CSV 로더가 영문·한국어 형식을 모두 읽는다.
- **FinanceDataReader `INVESTING:` 소스**
  - 검색은 `api.investing.com/api/search/v2/search`, 시세는 `iappapi.investing.com/get_screen.php`(모바일 앱 API)를 쓴다(소스에서 확인).
  - 비공식 경로라 언제든 막힐 수 있다.
- **investpy**: 2022년 Cloudflare 차단 이후 사실상 작동하지 않는다. **investiny**도 비공식이라 불안정하다.
- 주의
  - 가격이 **배당 미조정**이다.
  - 자동 수집은 이용약관상 제한될 수 있다. 개인 연구용으로만 쓸 것.

### 2. Yahoo Finance (yfinance / FinanceDataReader 기본 소스)
- 비공식 `query1/2.finance.yahoo.com/v8/finance/chart` API다. 키가 필요 없다.
- **수정주가(배당·분할 반영)**를 준다. QQQ는 1999년 상장 시점부터 받을 수 있다.
- 단점
  - 비공식이라 요청 제한(429)과 차단이 잦다.
  - 쿠키와 crumb 방식이 바뀌면 라이브러리 업데이트가 필요하다.
- FinanceDataReader는 미국 심볼을 기본적으로 Yahoo로 보낸다(`VIX`→`^VIX`, `IXIC`→`^IXIC`, `US10YT`→`^TNX` 매핑 확인).

### 3. FinanceDataReader (한국 라이브러리, 추천)
- `fdr.DataReader('QQQ')` 한 줄로 쓴다. 한국 사용자에게 가장 편하다.
- 소스를 통합해 준다: Yahoo(미국), 네이버(한국), KRX, FRED, Investing(모바일 API), 한국은행 ECOS.
- 반환 형식이 `Open/High/Low/Close/Volume`이라 이 저장소와 바로 맞는다(`scoring/sources.py`).

### 4. FRED (미 세인트루이스 연준)
- `fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10`은 **키 없이** CSV를 받을 수 있다(FinanceDataReader가 이 방식을 쓴다).
- 공식 API는 무료 키가 필요하다.
- 쓸 만한 시리즈
  - `DGS10` 10년물 금리
  - `BAMLH0A0HYM2` 하이일드 스프레드
  - `VIXCLS` VIX 종가
  - `NASDAQ100` 나스닥100 지수 종가
  - `NASDAQCOM` 나스닥 종합 종가
- 종가만 있고 고가·저가·거래량은 없다. 매크로 입력용으로 쓴다.

### 5. CBOE
- `cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv`: VIX 일봉 공식 무료 CSV다(1990~).
- 이 저장소 `--vix` 옵션에 파일 경로로 넣으면 된다.

### 6. 키가 필요한 공식 API (무료 등급 있음)
| 서비스 | 무료 등급 특징 | 비고 |
|---|---|---|
| **Tiingo** | 무료 키, 미국 EOD 수정주가 수십 년 | 품질과 안정성이 좋다. 비공식 소스가 막힐 때 1순위 |
| Alpha Vantage | 무료 키, 하루 호출 수 매우 적음 | 수정주가 일부 기능은 유료 |
| Polygon.io | 무료 플랜은 호출 수와 과거 기간 제한 | 유료 전환 시 틱 데이터까지 |
| Financial Modeling Prep / Twelve Data / EODHD | 무료 키, 호출 수 제한 | 해외 거래소 폭넓음 |
| Alpaca | 계좌 개설 시 무료 | 무료 데이터는 **IEX 거래소분 거래량**이라 VR·분산일 같은 거래량 지표가 왜곡된다 |
| Nasdaq.com | 비공식 JSON, 약 10년 | 헤더 요구가 까다롭다 |

### 7. 한국 시장
- **네이버 금융**: `fchart.stock.naver.com` 비공식이다. FinanceDataReader의 한국 종목 기본 소스다.
- **KRX 정보데이터시스템**: pykrx 1.2.9부터 **KRX 회원 로그인(`KRX_ID`, `KRX_PW`)이 필수**다(패키지 설명서에서 확인).
- **한국투자증권 KIS Developers**: 공식 Open API다. 계좌와 앱키가 필요하고, 국내·해외주식 일봉을 모두 준다. 가장 안정적인 공식 경로다.
- **공공데이터포털 금융위원회 주식시세정보**: 공식이고 무료 키다. 국내만 되고 하루 늦게 반영된다.

## 데이터 품질 체크리스트
1. **거래량 출처를 하나로 통일한다.** VR과 분산일은 거래량 합계 방식(통합 vs 단일 거래소)에 민감하다. Investing.com·Yahoo는 통합 거래량이고, Alpaca 무료 등급은 IEX뿐이다.
2. **수정주가 여부를 확인한다.** Investing.com은 미조정이고 Yahoo는 조정·미조정 둘 다 준다. 지표 계산은 어느 쪽이든 괜찮지만, 수익률 평가는 수정주가가 정확하다. 분할이 있었던 종목은 **반드시 수정주가**를 써야 한다.
3. **휴장일이 다른 벤치마크와 외부 데이터**는 직전 값으로 채운다(최대 5영업일, `scoring/score.py`의 `_align`).
4. **발표 시차**: FRED 일별 시리즈는 다음 날 공개되는 경우가 많다. 이 저장소는 하루 늦춰 쓴다.

## 이 클라우드 세션에서 직접 받으려면
- 현재 세션의 네트워크 정책은 PyPI와 GitHub만 허용한다. 위 사이트는 모두 차단(403)되어 있다.
- 세션의 환경 설정 → Network access에서 접근 수준을 넓히거나, `query2.finance.yahoo.com`, `fred.stlouisfed.org`, `api.investing.com`, `iappapi.investing.com`, `cdn.cboe.com` 등을 허용 도메인에 추가하면 된다.
- 로컬 PC에서는 별도 설정 없이 동작한다.
