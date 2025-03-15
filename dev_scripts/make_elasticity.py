from __future__ import annotations

from monty.serialization import dumpfn
from pymatgen.ext.matproj import MPRester

mpr = MPRester()

docs = mpr.elasticity.search()
docs_dict = [dict(d) for d in docs]
for d in docs_dict:
    for f in ["builder_meta", "state", "fields_not_requested", "nsites", "elements", "nelements", "composition",
        "composition_reduced", "formula_anonymous", "symmetry", "property_name", "deprecated", "deprecation_reasons",
        "last_updated", "origins", "warnings", "fitting_data"
    ]:
        try:
            del d[f]
        except:
            pass

new_docs = []

for d in docs_dict:
    for k, v in d.items():
        if (not isinstance(v, (str, float, int, list))) and v is not None:
            try:
                d[k] = v.as_dict()
            except:
                d[k] = dict(v)

    if d["bulk_modulus"] is not None:
        d["mp_id"] = d["material_id"]
        d["bulk_modulus_vrh"] = d["bulk_modulus"]["vrh"]
        d["shear_modulus_vrh"] = d["shear_modulus"]["vrh"]
        d["formula"] = d["formula_pretty"]

        del d["material_id"]
        del d["formula_pretty"]
        del d["order"]
        del d["fitting_method"]
        del d["young_modulus"]
        del d["bulk_modulus"]
        del d["shear_modulus"]


        new_docs.append(d)

# There are some wrong values in MP, e.g., bulk moduli exceeding 100000. We will clean the data
# removing nonsensical values.

print(f"Downloaded number of documents: {len(new_docs)}.")

new_docs = [d for d in new_docs if d["bulk_modulus_vrh"] < 500 and d["shear_modulus_vrh"] < 500 ]

print(f"Final cleaned number of documents: {len(new_docs)}.")

dumpfn(new_docs, "mp-elasticity-2025.3.json.gz")
