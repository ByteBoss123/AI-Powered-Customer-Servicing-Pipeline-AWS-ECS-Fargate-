"""Maps each of the 77 Banking77 intents to the support team that would own it.

The mapping is this project's design (documented here so it can be reviewed), not part of the dataset.
"Fraud & security" is the urgent queue: a missed message there can mean a customer keeps losing money.
"""

TEAMS = {
    "Fraud & security": [
        "card_payment_not_recognised", "cash_withdrawal_not_recognised", "compromised_card",
        "direct_debit_payment_not_recognised", "lost_or_stolen_card", "lost_or_stolen_phone",
        "transaction_charged_twice", "card_swallowed",
    ],
    "Cards": [
        "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "card_not_working", "change_pin", "contactless_not_working",
        "declined_card_payment", "disposable_card_limits", "get_disposable_virtual_card", "get_physical_card",
        "getting_spare_card", "getting_virtual_card", "order_physical_card", "pin_blocked",
        "virtual_card_not_working", "visa_or_mastercard", "pending_card_payment", "reverted_card_payment?",
        "supported_cards_and_currencies", "atm_support", "declined_cash_withdrawal", "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
    ],
    "Transfers & refunds": [
        "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer", "declined_transfer",
        "failed_transfer", "pending_transfer", "receiving_money", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing", "Refund_not_showing_up", "request_refund",
    ],
    "Top-ups": [
        "automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up", "top_up_by_cash_or_cheque",
        "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
    ],
    "Fees & exchange": [
        "card_payment_fee_charged", "card_payment_wrong_exchange_rate", "cash_withdrawal_charge", "exchange_charge",
        "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "fiat_currency_support",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "transfer_fee_charged",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "Account & identity": [
        "age_limit", "country_support", "edit_personal_details", "passcode_forgotten", "terminate_account",
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "why_verify_identity",
    ],
}

URGENT_TEAM = "Fraud & security"
INTENT_TO_TEAM = {intent: team for team, intents in TEAMS.items() for intent in intents}
