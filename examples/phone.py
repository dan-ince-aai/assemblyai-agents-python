"""Put a deployed agent on a phone number.

    export ASSEMBLYAI_API_KEY=...
    python examples/phone.py $AGENT_ID --country US --area-code 415 --yes

Buying a number is billable, so nothing happens without ``--yes``. Deploy the
agent with ``PUBLIC_BASE_URL`` set first: a phone call has no connected client,
so only tools with an ``http=`` config can run, and the SDK refuses to attach a
number to a declaration that still holds client-resident tools.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from assemblyai_agents import Client
from assemblyai_agents.models.rest import (
    NumberType,
    PurchaseAvailablePhoneNumberRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_id")
    parser.add_argument("--country", default="US", help="ISO 3166-1 alpha-2")
    parser.add_argument("--area-code", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="actually purchase")
    args = parser.parse_args()

    client = Client()
    print("numbers already on the account:")
    for number in client.phone_numbers.list():
        print(f"  {number.phone_number}  agent={number.agent_id}")

    request = PurchaseAvailablePhoneNumberRequest(
        country_code=args.country,
        number_type=NumberType.local,
        area_code=args.area_code,
        agent_id=args.agent_id,
    )
    if not args.yes:
        print("\nwould purchase:", request.model_dump(exclude_none=True))
        print("re-run with --yes to buy it")
        return

    number = client.phone_numbers.purchase_available(request)
    print(f"\npurchased {number.phone_number}; calls now reach agent {number.agent_id}")


if __name__ == "__main__":
    main()
