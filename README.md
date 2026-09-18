# Quant Strategy A — GitHub Actions + Telegram

최종 후보 A를 실전 신호로 변환하는 최소 구성입니다.

## 최종 Cap
- QQQ 90%
- TLT 40%
- GLD 80%
- XLE 30%

원시 목표비중 = `자산 Cap × 자산 신호강도`.
원시 목표비중 합계가 100%를 넘으면 비례 축소하고, 100% 이하면 나머지는 현금으로 둡니다.

## 자산별 신호

### QQQ — 일봉/주봉 50:50
- 일봉: MA40/100/200, 공통 Buy +1%, Sell -3%, C
- 주봉: MA40/100/200, 공통 Buy +0%, Sell -3%, C
- 최종 신호강도 = 두 신호의 평균

### TLT — 일봉/주봉 50:50
- 일봉: MA10/70/180, 개별 밴드 (4/2, 1.5/2, 2/1.5)%, C
- 주봉: MA10/40/160, 개별 밴드 (3/0, 4/0, 0/3.5)%, D
- 최종 신호강도 = 두 신호의 평균

### GLD — 일봉
- MA10/40/170
- 밴드 (0/1, 1.5/1, 1.5/2)%
- C

### XLE — Min
- 일봉: MA20/160/200, 밴드 (0/4, 0.5/2, 0/1)%, C
- 주봉: MA50/80/110, 공통 3.5/3.5%, D
- 최종 신호강도 = 일봉/주봉 중 낮은 값

## C / D 정의

C:
- 신호 개수가 1개 늘거나 줄면 비중 ±50%p
- 하루에 2개 변화하면 ±100%p까지 즉시 반영

D:
- 신호가 변한 날에만 비중 조정
- 신호 변화량과 관계없이 하루 최대 ±50%p
- 신호 개수가 변하지 않으면 비중도 유지

## GitHub Secrets
Repository > Settings > Secrets and variables > Actions 에 추가:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## GitHub Actions 시간
`30 21 * * 0-4` (UTC) = 한국시간 월~금 오전 6:30.
월요일 아침에는 직전 금요일 미국장 종가 신호를 사용합니다.

## 먼저 로컬 CSV로 검증
현재 업로드된 CSV가 있는 폴더에서:

```bash
python strategy.py --local-dir /path/to/csvs
```

실전 데이터 조회만 테스트:

```bash
python strategy.py
```

텔레그램까지 전송:

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python strategy.py --send
```

## 중요한 실전 체크
1. 텔레그램 목표비중이 백테스트 마지막 신호와 일치하는지 확인
2. 최소 1~2주 동안 실제 주문 없이 페이퍼 트레이딩
3. 한국 상장 ETF 종목코드는 `config.yaml`의 `korea_mapping`에 입력
4. 한국 휴장일 및 매매 가능 여부 처리는 다음 단계에서 추가
