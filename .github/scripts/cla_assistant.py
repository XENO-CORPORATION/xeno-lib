import base64
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


MARKER = "<!-- xeno-cla -->"


def github_api(method, path, token, payload=None):
    url = f"https://api.github.com{path}"
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "xeno-lib-cla",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code} {body}") from error


def github_api_or_none(method, path, token):
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "xeno-lib-cla",
    }
    request = urllib.request.Request(url, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code} {body}") from error


def matches_allowlist(value, patterns):
    if not value:
        return False
    lowered = value.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def get_pr_number(event_name, event):
    if event_name == "pull_request_target":
        return event["pull_request"]["number"]
    if event_name == "issue_comment" and event.get("issue", {}).get("pull_request"):
        return event["issue"]["number"]
    return None


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


def load_ledger(owner, repo, branch, path, token):
    encoded_path = urllib.parse.quote(path, safe="/")
    response = github_api_or_none(
        "GET",
        f"/repos/{owner}/{repo}/contents/{encoded_path}?ref={urllib.parse.quote(branch, safe='')}",
        token,
    )

    if response is None:
        return {"signedContributors": []}, None

    content = base64.b64decode(response["content"]).decode("utf-8")
    ledger = json.loads(content)
    ledger.setdefault("signedContributors", [])
    return ledger, response["sha"]


def store_ledger(owner, repo, branch, path, token, ledger, sha, signer, pr_number):
    encoded_path = urllib.parse.quote(path, safe="/")
    content = json.dumps(ledger, indent=2).encode("utf-8")
    payload = {
        "message": f"cla: record signature for @{signer['login'] or signer['name']} on PR #{pr_number}",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    github_api("PUT", f"/repos/{owner}/{repo}/contents/{encoded_path}", token, payload)


def evaluate(committers, ledger):
    signed_ids = {entry["id"] for entry in ledger.get("signedContributors", []) if entry.get("id")}
    signed = []
    unsigned = []
    unknown = []

    for committer in committers:
        if not committer.get("id"):
            unknown.append(committer)
        elif committer["id"] in signed_ids:
            signed.append(committer)
        else:
            unsigned.append(committer)

    return signed, unsigned, unknown


def render_comment(document_url, sign_text, signed, unsigned, unknown):
    if not unsigned and not unknown:
        return (
            f"{MARKER}\n"
            "All contributors on this pull request have signed the CLA. Thank you."
        )

    lines = [
        MARKER,
        "Thank you for your contribution. Before we can merge this PR, each committer on this pull request must agree to our "
        f"[Contributor License Agreement]({document_url}).",
        "",
        "To sign, reply with this exact text:",
        "",
        f"> {sign_text}",
    ]

    if unsigned:
        lines.extend(["", "Unsigned committers:"])
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
                "Add the commit email address to the matching GitHub account, then post the CLA signature comment again.",
            ]
        )

    if signed:
        lines.extend(["", f"Already signed: {len(signed)} committer(s)."])

    return "\n".join(lines)


def upsert_comment(owner, repo, pr_number, token, body):
    comments = github_api(
        "GET",
        f"/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100",
        token,
    )

    existing = None
    for comment in comments:
        if comment.get("user", {}).get("login") == "github-actions[bot]" and MARKER in comment.get(
            "body", ""
        ):
            existing = comment
            break

    if existing:
        github_api(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/comments/{existing['id']}",
            token,
            {"body": body},
        )
    else:
        github_api(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            token,
            {"body": body},
        )


def try_upsert_comment(owner, repo, pr_number, token, body):
    try:
        upsert_comment(owner, repo, pr_number, token, body)
    except RuntimeError as error:
        print(f"Warning: unable to update CLA comment: {error}", file=sys.stderr)
        return False
    return True


def rerun_latest_pr_check(owner, repo, pr_number, token):
    runs = github_api(
        "GET",
        "/repos/{owner}/{repo}/actions/workflows/cla.yml/runs?event=pull_request_target&per_page=20".format(
            owner=owner,
            repo=repo,
        ),
        token,
    )

    for run in runs.get("workflow_runs", []):
        pull_requests = run.get("pull_requests", [])
        if any(pr.get("number") == pr_number for pr in pull_requests):
            try:
                github_api("POST", f"/repos/{owner}/{repo}/actions/runs/{run['id']}/rerun", token, {})
            except RuntimeError:
                return
            return


def maybe_record_signature(event_name, event, committers, unsigned, ledger, ledger_sha, owner, repo, token):
    if event_name != "issue_comment":
        return ledger, False

    comment = event.get("comment", {})
    body = (comment.get("body") or "").strip()
    if body != os.environ["CLA_SIGN_TEXT"]:
        return ledger, False

    commenter = comment.get("user", {})
    commenter_id = commenter.get("id")
    if not commenter_id:
        return ledger, False

    unsigned_ids = {committer["id"] for committer in unsigned if committer.get("id")}
    if commenter_id not in unsigned_ids:
        return ledger, False

    if any(entry.get("id") == commenter_id for entry in ledger.get("signedContributors", [])):
        return ledger, False

    signer = next(committer for committer in committers if committer.get("id") == commenter_id)
    ledger["signedContributors"].append(
        {
            "name": signer["login"] or signer["name"],
            "id": signer["id"],
            "comment_id": comment["id"],
            "created_at": comment["created_at"],
            "repoId": event["repository"]["id"],
            "pullRequestNo": event["issue"]["number"],
        }
    )
    store_ledger(
        owner,
        repo,
        os.environ["CLA_LEDGER_BRANCH"],
        os.environ["CLA_LEDGER_PATH"],
        token,
        ledger,
        ledger_sha,
        signer,
        event["issue"]["number"],
    )
    return ledger, True


def main():
    token = os.environ["GITHUB_TOKEN"]
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/", 1)
    event_name = os.environ["GITHUB_EVENT_NAME"]
    allowlist = [entry.strip() for entry in os.environ["CLA_ALLOWLIST"].split(",") if entry.strip()]

    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
        event = json.load(handle)

    pr_number = get_pr_number(event_name, event)
    if not pr_number:
        print("No pull request context for this event.")
        return 0

    committers = list_pr_commits(owner, repo, pr_number, token, allowlist)
    ledger, ledger_sha = load_ledger(
        owner,
        repo,
        os.environ["CLA_LEDGER_BRANCH"],
        os.environ["CLA_LEDGER_PATH"],
        token,
    )

    signed, unsigned, unknown = evaluate(committers, ledger)
    ledger, recorded = maybe_record_signature(
        event_name,
        event,
        committers,
        unsigned,
        ledger,
        ledger_sha,
        owner,
        repo,
        token,
    )

    signed, unsigned, unknown = evaluate(committers, ledger)
    try_upsert_comment(
        owner,
        repo,
        pr_number,
        token,
        render_comment(
            os.environ["CLA_DOCUMENT_URL"],
            os.environ["CLA_SIGN_TEXT"],
            signed,
            unsigned,
            unknown,
        ),
    )

    if event_name == "issue_comment":
        if recorded:
            rerun_latest_pr_check(owner, repo, pr_number, token)
        return 0

    if unsigned or unknown:
        print("Unsigned committers remain.")
        return 1

    print("All contributors have signed the CLA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
