set THRESH=8
if not exist qc mkdir qc

for /f usebackq %%I in ("ids.txt") do (
  echo ==== %%I ====
  python validate_normals_and_theta.py --boxes output\%%I_boxes.pdb --src   input_data\%%I.pdb --out-prefix qc\%%I --angle-thresh %THRESH%
)