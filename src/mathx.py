
import re
BOXED = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


MAX_ABS = 10**15   # 이보다 큰 값은 파싱 오류로 간주 (대회 정답은 정상 범위)

def _bound(v):
    if v is None: return None
    try:
        if abs(int(v)) > MAX_ABS: return None
    except Exception:
        return None
    return int(v)

def _to_int_raw(tok):
    if tok is None: return None
    s = str(tok)
    for a, b in [("\\!",""),("\\,",""),("\\;",""),("\\ ",""),
                 ("\u2212","-"),("\u2013","-"),("\u2014","-"),
                 ("$",""),("%",""),("\\%",""),("{",""),("}",""),
                 ("*",""),("`",""),('"',""),("'","")]:
        s = s.replace(a, b)
    s = s.strip().strip(".").strip(":").strip()
    s = s.replace(" ", "").replace(",", "")
    if s in ("", "-", "+"): return None
    m = re.fullmatch(r"\\[dt]?frac(-?\d+)(-?\d+)", s)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n // d if d and n % d == 0 else None
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n // d if d and n % d == 0 else None
    if re.fullmatch(r"[-+]?\d+", s): return int(s)
    if re.fullmatch(r"[-+]?\d*\.\d+", s):
        f = float(s)
        return int(round(f)) if abs(f - round(f)) < 1e-6 else None
    nums = re.findall(r"[-+]?\d+", s)
    return int(nums[0]) if len(nums) == 1 else None

def end_kind(s):
    s = (s or "").rstrip()
    if not s: return "empty"
    if re.search(r"FINAL[_\s]*ANSWER\s*[:\uff1a]?\s*\\?\W{0,3}-?\d", s[-90:], re.I):
        return "clean_final"
    if s[-1] in ".!?}$": return "sentence_end"
    if s[-1].isdigit(): return "ends_digit"
    return "mid_cut"

def extract(text, mode="lenient"):
    """mode: 'lenient' (채점/제출) | 'strict' (학습데이터 채택)
       return (value|None, method)"""
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    t = text
    ms = list(re.finditer(r"FINAL[_\s]*ANSWER\s*[:\uff1a=]?\s*(.{0,60})", t, re.I))
    for m in reversed(ms):
        seg = m.group(1).split("\n")[0]
        v = to_int(seg)
        if v is None:
            nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", seg)
            if nums: v = to_int(nums[0])
        if v is not None: return v, "final_answer"
    for b in reversed(BOXED.findall(t)):
        v = to_int(b)
        if v is None:
            nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", b)
            if nums: v = to_int(nums[-1])
        if v is not None: return v, "boxed"
    if mode == "strict":
        return None, "strict_reject"
    for m in reversed(list(re.finditer(
            r"(?:answer|result|total)\s*(?:is|:|=)\s*(.{0,40})", t, re.I))):
        seg = m.group(1).split("\n")[0]
        v = to_int(seg)
        if v is None:
            nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", seg)
            if nums: v = to_int(nums[0])
        if v is not None: return v, "answer_is"
    tail = "\n".join([l for l in t.strip().split("\n") if l.strip()][-3:])
    for n in reversed(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)):
        v = to_int(n)
        if v is not None: return v, "last_number"
    return None, "fail"

def sft_acceptable(text):
    """SFT 학습 타깃으로 써도 되는가 (절단/형식 불량 배제)"""
    v, m = extract(text, mode="strict")
    return (v is not None) and (end_kind(text) == "clean_final"), v

PROMPT_SYSTEM = "You are a helpful assistant."
PROMPT_USER = ("Solve the math problem carefully. The answer is guaranteed to be an "
               "integer. Write the final answer on the last line exactly as "
               "FINAL_ANSWER: <integer>. Do not write anything after that line.\n\n"
               "Problem: {q}")


def to_int(tok):
    return _bound(_to_int_raw(tok))
