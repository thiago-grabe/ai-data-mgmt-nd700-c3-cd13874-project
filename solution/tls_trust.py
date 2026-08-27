"""Use the operating system's trust store for TLS instead of certifi's bundle.

Why this exists: on a network that performs TLS interception (a corporate proxy
re-signing traffic with its own CA), downloading the CLIP weights from
huggingface.co fails with

    SSLCertVerificationError: unable to get local issuer certificate

The proxy's root CA is installed in the OS keychain — which is why browsers and
`az` work — but Python's `requests`/`urllib3` stack verifies against the certifi
bundle shipped inside the virtualenv, which has never heard of it.

`truststore` redirects verification to the OS trust store. That is strictly more
correct than a vendored snapshot: it honours the CAs the machine's administrator
actually installed, and it honours revocations. It is emphatically NOT the same as
disabling verification — certificates are still validated, just against the right
root set.

Safe to call unconditionally: it no-ops if truststore is unavailable, and Azure
SDK calls work either way (they already succeed against certifi here).
"""

def enable() -> bool:
    """Route TLS verification through the OS trust store. Returns True if applied."""
    try:
        import truststore
    except ImportError:
        return False

    try:
        truststore.inject_into_ssl()
        return True
    except Exception as e:  # pragma: no cover - defensive, never fatal
        print(f"[tls] could not enable the OS trust store ({type(e).__name__}); "
              "falling back to certifi. If model downloads fail with "
              "CERTIFICATE_VERIFY_FAILED, see docs/AZURE_SETUP.md.")
        return False
