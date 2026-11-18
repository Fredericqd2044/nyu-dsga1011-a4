import pickle, re, sys
from collections import Counter

MODEL = sys.argv[1] if len(sys.argv) > 1 else "experiment"
nl   = [l.strip() for l in open("data/dev.nl")]
gold = [l.strip() for l in open("data/dev.sql")]
pred = [l.strip() for l in open(f"results/t5_ft_{MODEL}_dev.sql")]

def load(path):
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0], obj[1]
    return obj, [""] * len(obj)

g_rec, _ = load("records/ground_truth_dev.pkl")
p_rec, p_err = load(f"records/t5_ft_{MODEL}_dev.pkl")
print(f"model={MODEL}  examples: nl={len(nl)} gold={len(gold)} pred={len(pred)} records={len(p_rec)}")

def skeleton(q):
    q = re.sub(r"'[^']*'", "'VAL'", q)
    q = re.sub(r"\b\d+\b", "NUM", q)
    return " ".join(q.split()).lower()

def tables(q):
    out = set()
    for frm in re.findall(r"from\s+(.*?)(?:\bwhere\b|\)|$)", q.lower()):
        for part in frm.split(","):
            if part.strip():
                out.add(part.split()[0])
    return out

def same_records(a, b):
    try:
        return set(map(tuple, a)) == set(map(tuple, b))
    except TypeError:
        return a == b

cats, examples, err_msgs = Counter(), {}, Counter()
for i in range(len(gold)):
    if p_err[i]:
        cat = "1. SQL execution error"
        err_msgs[str(p_err[i]).split(":")[0][:60]] += 1
    elif same_records(p_rec[i], g_rec[i]):
        cat = "0. correct records"
    elif skeleton(pred[i]) == skeleton(gold[i]):
        cat = "2. right structure, wrong values"
    elif tables(pred[i]) != tables(gold[i]):
        cat = "3. wrong tables / joins"
    else:
        cat = "4. same tables, wrong/missing conditions"
    cats[cat] += 1
    examples.setdefault(cat, i)

lt = sum(1 for i in range(len(gold)) if "<" in gold[i])
lt_wrong = sum(1 for i in range(len(gold)) if "<" in gold[i] and not same_records(p_rec[i], g_rec[i]))
print("\n=== ERROR CATEGORIES (dev, n=%d) ===" % len(gold))
for c in sorted(cats):
    print(f"{c:45s} {cats[c]:4d}  ({100*cats[c]/len(gold):.1f}%)")
print(f"\ngold queries containing '<': {lt}, of which wrong: {lt_wrong}")
print("\n=== MOST COMMON SQL ERROR MESSAGES ===")
for m, n in err_msgs.most_common(5):
    print(f"{n:4d}  {m}")
for c in sorted(examples):
    if c.startswith("0"):
        continue
    i = examples[c]
    print(f"\n=== EXAMPLE: {c} (line {i+1}) ===")
    print("Question :", nl[i])
    print("Gold SQL :", gold[i][:400])
    print("Model SQL:", pred[i][:400])
    if p_err[i]:
        print("Error    :", str(p_err[i])[:200])
