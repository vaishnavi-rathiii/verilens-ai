from src.ai.claim_extractor import extract_claims


article = """
The Indian Space Research Organisation successfully launched
Chandrayaan-3 on July 14, 2023, from the Satish Dhawan Space Centre.
The spacecraft entered lunar orbit on August 5, 2023.
The Vikram lander successfully soft-landed near the Moon's south
polar region on August 23, 2023.
India became the first country to successfully land a spacecraft
near the Moon's south polar region.
"""


claims = extract_claims(article)

print("\nEXTRACTED CLAIMS:\n")

for i, item in enumerate(claims, start=1):
    print(f"{i}. {item.get('claim')}")
    print(f"   Importance: {item.get('importance')}")
    