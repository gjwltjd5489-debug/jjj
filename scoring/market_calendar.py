"""NYSE 휴장일과 '마지막으로 끝난 정규장' 계산 (외부 패키지 없이 규칙 기반).

정기 휴장일만 다룬다. 국가 애도일 같은 비정기 휴장은 여기서 모르므로 '데이터 지연'으로 보고된다.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from dateutil.easter import easter

NY = ZoneInfo("America/New_York")
SESSION_DONE = time(16, 30)  # 정규장 16:00 마감 + 데이터 반영 여유


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> dict[date, str]:
    h: dict[date, str] = {}
    ny = date(year, 1, 1)
    if ny.weekday() != 5:  # 토요일 신정은 전날(12/31)로 당기지 않는다
        h[_observed(ny)] = "신정"
    h[_nth_weekday(year, 1, 0, 3)] = "마틴 루서 킹 주니어 데이"
    h[_nth_weekday(year, 2, 0, 3)] = "대통령의 날"
    h[easter(year) - timedelta(days=2)] = "성금요일"
    h[_last_weekday(year, 5, 0)] = "메모리얼 데이"
    if year >= 2022:
        h[_observed(date(year, 6, 19))] = "준틴스"
    h[_observed(date(year, 7, 4))] = "독립기념일"
    h[_nth_weekday(year, 9, 0, 1)] = "노동절"
    h[_nth_weekday(year, 11, 3, 4)] = "추수감사절"
    h[_observed(date(year, 12, 25))] = "성탄절"
    return h


def holiday_name(d: date) -> str | None:
    return nyse_holidays(d.year).get(d)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and holiday_name(d) is None


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def session_status(now: datetime | None = None) -> dict:
    """지금 시점에서 기대하는 '마지막으로 끝난 정규장'과 휴장 여부.

    반환: target(마지막 정규장 날짜), checked(확인 대상 날짜), holiday(휴장 이름 또는 None),
          next_open(다음 거래일)
    - 뉴욕 기준 16:30 이전이면 전날을 확인 대상으로 본다.
    - 확인 대상이 평일 휴장일이면 holiday 에 이름이 들어가고 target 은 그 전 거래일이다.
    """
    now = (now or datetime.now(tz=NY)).astimezone(NY)
    checked = now.date() if now.time() >= SESSION_DONE else now.date() - timedelta(days=1)
    name = holiday_name(checked) if checked.weekday() < 5 else None
    target = checked if is_trading_day(checked) else previous_trading_day(checked)
    return {"target": target, "checked": checked, "holiday": name, "next_open": next_trading_day(checked)}
