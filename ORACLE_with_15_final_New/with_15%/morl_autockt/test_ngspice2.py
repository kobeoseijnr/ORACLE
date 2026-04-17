"""Test NGSpice via os.system (same as wrapper)."""
import os

ngspice = r"C:\Users\kobeo\Downloads\ngspice-45.2_64\Spice64\bin\ngspice_con.exe"
netlist = r"C:\Users\kobeo\AppData\Local\Temp\ckt_da\designs_two_stage_opamp\two_stage_opamp_34_34_34_34_34_15_2.1e-125560\two_stage_opamp_34_34_34_34_34_15_2.1e-12.cir"

# Delete existing csv files first
folder = os.path.dirname(netlist)
for f in ['ac.csv', 'dc.csv']:
    p = os.path.join(folder, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Deleted {p}")

# Run exactly as wrapper does
fpath_fwd = netlist.replace("\\", "/")
command = '"{cmd}" -b "{netlist}" > NUL 2>&1'.format(cmd=ngspice, netlist=fpath_fwd)
print(f"Command: {command}")
exit_code = os.system(command)
print(f"Exit code: {exit_code}")

# Check output
ac = os.path.join(folder, "ac.csv")
dc = os.path.join(folder, "dc.csv")
print(f"ac.csv exists: {os.path.exists(ac)} (size: {os.path.getsize(ac) if os.path.exists(ac) else 0})")
print(f"dc.csv exists: {os.path.exists(dc)} (size: {os.path.getsize(dc) if os.path.exists(dc) else 0})")

if not os.path.exists(ac):
    # Try without NUL redirect to see error
    print("\nRetrying without redirect...")
    command2 = '"{cmd}" -b "{netlist}"'.format(cmd=ngspice, netlist=fpath_fwd)
    exit_code2 = os.system(command2)
    print(f"Exit code: {exit_code2}")
    print(f"ac.csv exists now: {os.path.exists(ac)}")
