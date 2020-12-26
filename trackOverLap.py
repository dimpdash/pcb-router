from svgwrite import

dwg = svgwrite.Drawing('test.svg', profile='tiny', size = (8000,8000))

for track in pcbTracks:
    p1 = (track[3],track[4])
    p2 = (track[5], track[6])
    width = track[0]

    dwg.add(dwg.line(start=p1, end=p2, stroke_width = width, stroke="black"))
    dwg.add(dwg.circle(center=p1, r = width/2))
    dwg.add(dwg.circle(center=p2, r = width/2))

dwg.save()

SVG('test.svg')