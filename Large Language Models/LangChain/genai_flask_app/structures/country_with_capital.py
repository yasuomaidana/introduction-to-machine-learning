from pydantic import BaseModel


class CountryWithCapital(BaseModel):
    country: str
    capital: str
