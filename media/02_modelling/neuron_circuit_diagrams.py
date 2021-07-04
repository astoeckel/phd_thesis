import schemdraw
import schemdraw.elements as elm


d = schemdraw.Drawing(unit=2.0, inches_per_unit=.5, lw=1.0, fontsize=9.0)
d += elm.SourceI().label("$J(t)$").right()

d += (N1 := elm.Dot())
d += elm.Resistor().label("$g_\\mathrm{L}$").down()
d += elm.SourceV().label("$E_\\mathrm{L}$").reverse().down()

d += elm.Line().length(d.unit*1.25).at(N1.end).right()
d += (N2 := elm.Dot())
d += elm.Switch(action="close").label("Reset\nmechanism", loc='bottom').down()
d += elm.SourceV().label("$v_\\mathrm{reset}$").reverse().down()
d += elm.Dot()
d += elm.Ground()
d += elm.Line().length(d.unit*1.25).left()

d += elm.Line().length(d.unit*1.25).at(N2.end).right()
d += elm.Line().down().length(d.unit*0.5)
d += elm.Capacitor().label("$C_\\mathrm{m}$")
d += elm.Line().length(d.unit*0.5)
d += elm.Line().length(d.unit*1.25).left()

fig, ax = plt.subplots(figsize=(4.0, 4.0))
ax.set_aspect(1)
ax.set_clip_on(False)
ax.set_xmargin(0.1)
ax.set_ymargin(0.1)
d.draw(ax=ax, show=False, showframe=False)
utils.remove_frame(ax)

utils.save(fig)
