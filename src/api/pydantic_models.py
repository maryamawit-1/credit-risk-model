from pydantic import BaseModel

class PredictionRequest(BaseModel):
    num__Amount: float
    num__CountryCode: float
    num__total_transaction_amount: float
    num__avg_transaction_amount: float
    num__transaction_count: float
    num__std_transaction_amount: float
    num__transaction_hour: float
    num__transaction_day: float
    num__transaction_month: float
    num__transaction_year: float

    cat__0: float
    cat__1: float
    cat__2: float
    cat__3: float
    cat__4: float
    cat__5: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int