"""Run every experiment and save all figures to ../outputs."""
import equilibria, jacobian, bifurcation, rk4, fields, phase_plane
import schematic_plots
bifurcation.make_figs(); rk4.make_figs(); fields.make_figs(); phase_plane.portraits(); phase_plane.limit_cycle(3.0)
schematic_plots.make()
print("all figures written to ../outputs")
