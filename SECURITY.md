# Security Policy

## Supported Versions

The following versions of IntentLog are currently receiving security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.

2. **Email the maintainers** with details about the vulnerability:
   - Create a private security advisory on GitHub: [Report a vulnerability](https://github.com/kase1111-hash/IntentLog/security/advisories/new)
   - Or email the project maintainers directly

3. **Include the following information**:
   - Type of vulnerability (e.g., injection, cryptographic weakness, path traversal)
   - Location of the affected source code (file, line number)
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if available)
   - Impact assessment and potential attack scenarios

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Initial Assessment**: Within 7 days, we will provide an initial assessment of the vulnerability.
- **Resolution Timeline**: We aim to resolve critical vulnerabilities within 30 days.
- **Disclosure**: We will coordinate with you on public disclosure timing.

### Security Update Process

1. Security patches are released as soon as possible after verification.
2. Critical vulnerabilities may trigger immediate patch releases.
3. Security advisories are published on GitHub after fixes are available.
4. The CHANGELOG.md documents security-related changes.

## Security Best Practices for Users

### Installation

```bash
# Always install from trusted sources
pip install intentlog

# Verify package integrity
pip hash intentlog
```

### Cryptographic Features

- **Key Storage**: Private keys are stored with restricted permissions (0o600).
- **Key Rotation**: Regularly rotate signing keys for production deployments.
- **Password Protection**: Use password-protected private keys for sensitive applications.

```bash
# Generate a new keypair with password protection
ilog keys generate --password
```

### API Keys and Secrets

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- The `.gitignore` automatically excludes `*.key` and `*.secret` files

```bash
# Set API keys via environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Privacy and Data Protection

- Use appropriate privacy levels for sensitive intents
- Encrypt confidential content before storage
- Review access control policies regularly

```bash
# Encrypt sensitive intents
ilog privacy encrypt --level CONFIDENTIAL
```

## Known Security Considerations

### Current Limitations

1. **Local File Storage**: IntentLog stores data in local JSON files. Ensure proper file system permissions.
2. **Memory Handling**: API keys are held in memory during runtime. Consider this for security-critical deployments.
3. **Audit Logs**: Current audit logs do not have cryptographic integrity protection (planned for future release).

### Security Audit

A comprehensive security audit has been performed. See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for detailed findings and recommendations.

## Security Features

IntentLog includes several security features:

- **Ed25519 Digital Signatures**: Cryptographic signing of intents
- **Merkle Tree Chain Linking**: Tamper-evident history with SHA-256
- **Fernet Encryption**: AES-128-CBC encryption for sensitive content
- **Privacy Levels**: Granular access control (PUBLIC to TOP_SECRET)
- **Input Validation**: Protection against path traversal and injection attacks
- **Rate Limiting**: Protection against abuse of LLM APIs

## Responsible Disclosure

We appreciate the security research community and believe in responsible disclosure. Researchers who report valid vulnerabilities will be:

- Credited in the security advisory (unless anonymity is requested)
- Acknowledged in release notes
- Thanked for their contribution to making IntentLog more secure

## Contact

For security-related inquiries:

- GitHub Security Advisories: [Create Advisory](https://github.com/kase1111-hash/IntentLog/security/advisories/new)
- GitHub Issues (for non-security bugs): [Issues](https://github.com/kase1111-hash/IntentLog/issues)

---

*This security policy is effective as of January 2026 and applies to IntentLog v0.1.0 and later.*
