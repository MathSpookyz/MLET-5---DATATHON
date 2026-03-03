from pydantic import BaseModel


class PredictionRequest(BaseModel):
    INDE_FINAL: float
    IAA_FINAL: float
    IEG_FINAL: float
    IPS_FINAL: float
    IDA_FINAL: float
    IPP_FINAL: float
    IPV_FINAL: float
    IAN_FINAL: float


class PredictionResponse(BaseModel):
    Grupo: int
    Nivel: str
    Probabilidade_PV: float
