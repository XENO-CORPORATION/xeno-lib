import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request


def github_api(method, path, token):
    url = f"https://api.github.com{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xeno-lib-cla",
        },
        method=method,
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code} {body}") from error


def matches_allowlist(value, patterns):
    if not value:
        return False
    lowered = value.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def list_pr_commits(owner, repo, pr_number, token, allowlist):
    committers = []
    seen = set()
    page = 1

    while True:
        response = github_api(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/commits?per_page=100&page={page}",
            token,
        )
        if not response:
            break

        for commit in response:
            user = commit.get("author") or commit.get("committer")
            fallback = commit.get("commit", {}).get("author") or commit.get("commit", {}).get(
                "committer", {}
            )
            login = user.get("login") if user else None
            user_id = user.get("id") if user else None
            name = login or fallback.get("name") or "unknown"

            if matches_allowlist(login or name, allowlist):
                continue

            key = f"id:{user_id}" if user_id else f"name:{name.lower()}"
            if key in seen:
                continue
            seen.add(key)

            committers.append(
                {
                    "name": name,
                    "login": login,
                    "id": user_id,
                }
            )

        if len(response) < 100:
            break
        page += 1

    return committers


def list_pr_signatures(owner, repo, pr_number, token, sign_text):
    signed_ids = set()
    signed_logins = set()
    page = 1

    while True:
        comments = github_api(
            "GET",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100&page={page}",
            token,
        )
        if not comments:
            break

        for comment in comments:
            if (comment.get("body") or "").strip() != sign_text:
                continue

            user = comment.get("user") or {}
            if user.get("id"):
                signed_ids.add(user["id"])
            if user.get("login"):
                signed_logins.add(user["login"].lower())

        if len(comments) < 100:
            break
        page += 1

    return signed_ids, signed_logins


def evaluate(committers, signed_ids, signed_logins):
    signed = []
    unsigned = []
    unknown = []

    for committer in committers:
        if committer.get("id") and committer["id"] in signed_ids:
            signed.append(committer)
        elif committer.get("login") and committer["login"].lower() in signed_logins:
            signed.append(committer)
        elif committer.get("id") or committer.get("login"):
            unsigned.append(committer)
        else:
            unknown.append(committer)

    return signed, unsigned, unknown


def render_report(document_url, sign_text, signed, unsigned, unknown):
    lines = [
        "# CLA status",
        "",
        f"Document: {document_url}",
        "",
    ]

    if not unsigned and not unknown:
        lines.extend(
            [
                "All non-allowlisted contributors on this pull request have signed the CLA.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "Before this pull request can be merged, each non-allowlisted contributor must post this exact PR comment:",
            "",
            f"`{sign_text}`",
            "",
            "After the comment is posted, rerun the required `cla` check.",
        ]
    )

    if unsigned:
        lines.extend(["", "Unsigned contributors:"])
        for committer in unsigned:
            identity = f"@{committer['login']}" if committer.get("login") else committer["name"]
            lines.append(f"- {identity}")

    if unknown:
        lines.extend(
            [
                "",
                "Commits not linked to a GitHub account:",
            ]
        )
        for committer in unknown:
            lines.append(f"- {committer['name']}")
        lines.extend(
            [
                "",
                "Link the commit email address to the correct GitHub account, then rerun `cla`.",
            ]
        )

    if signed:
        lines.extend(["", f"Already signed: {len(signed)} contributor(s)."])

    return "\n".join(lines)


def write_step_summary(report):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n")


def main():
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request_target":
        print("CLA validation only runs on pull_request_target.")
        return 0

    token = os.environ["GITHUB_TOKEN"]
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/", 1)
    allowlist = [entry.strip() for entry in os.environ["CLA_ALLOWLIST"].split(",") if entry.strip()]

    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
        event = json.load(handle)

    pr_number = event["pull_request"]["number"]
    sign_text = os.environ["CLA_SIGN_TEXT"]
    document_url = os.environ["CLA_DOCUMENT_URL"]

    committers = list_pr_commits(owner, repo, pr_number, token, allowlist)
    signed_ids, signed_logins = list_pr_signatures(owner, repo, pr_number, token, sign_text)
    signed, unsigned, unknown = evaluate(committers, signed_ids, signed_logins)

    report = render_report(document_url, sign_text, signed, unsigned, unknown)
    print(report)
    write_step_summary(report)

    if unsigned or unknown:
        return 1

    print("All contributors have signed the CLA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
