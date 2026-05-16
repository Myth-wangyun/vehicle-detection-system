#!/usr/bin/env python3
"""Generate vehicle detection images programmatically"""

import base64
import io
import json

# We'll generate SVG-based images as data URLs
# These are real-looking traffic scenes with detection boxes

def generate_city_road():
    """City road with multiple vehicles"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <defs>
            <linearGradient id="sky1" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#3498db"/>
                <stop offset="100%" style="stop-color:#87CEEB"/>
            </linearGradient>
            <linearGradient id="road1" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#2c3e50"/>
                <stop offset="100%" style="stop-color:#34495e"/>
            </linearGradient>
        </defs>
        <!-- Sky -->
        <rect width="640" height="150" fill="url(#sky1)"/>
        <!-- Sun -->
        <circle cx="580" cy="50" r="30" fill="#f1c40f"/>
        <!-- Buildings -->
        <rect x="30" y="50" width="80" height="100" fill="#34495e"/>
        <rect x="130" y="70" width="60" height="80" fill="#3d566e"/>
        <rect x="210" y="40" width="90" height="110" fill="#2c3e50"/>
        <rect x="320" y="60" width="70" height="90" fill="#34495e"/>
        <rect x="410" y="30" width="100" height="120" fill="#3d566e"/>
        <rect x="530" y="50" width="80" height="100" fill="#2c3e50"/>
        <!-- Windows -->
        <g fill="#f4d03f" opacity="0.7">
            <rect x="40" y="60" width="15" height="15"/><rect x="65" y="60" width="15" height="15"/>
            <rect x="40" y="85" width="15" height="15"/><rect x="65" y="85" width="15" height="15"/>
            <rect x="140" y="80" width="12" height="12"/><rect x="160" y="80" width="12" height="12"/>
            <rect x="140" y="100" width="12" height="12"/><rect x="160" y="100" width="12" height="12"/>
            <rect x="420" y="40" width="15" height="15"/><rect x="445" y="40" width="15" height="15"/>
            <rect x="470" y="40" width="15" height="15"/><rect x="495" y="40" width="15" height="15"/>
            <rect x="420" y="65" width="15" height="15"/><rect x="445" y="65" width="15" height="15"/>
        </g>
        <!-- Road -->
        <rect x="0" y="150" width="640" height="250" fill="url(#road1)"/>
        <!-- Lane markings -->
        <line x1="0" y1="200" x2="640" y2="200" stroke="#fff" stroke-width="3" stroke-dasharray="40,20"/>
        <line x1="0" y1="250" x2="640" y2="250" stroke="#f39c12" stroke-width="2" stroke-dasharray="20,10"/>
        <line x1="0" y1="300" x2="640" y2="300" stroke="#f39c12" stroke-width="2" stroke-dasharray="20,10"/>
        <!-- Sidewalk -->
        <rect x="0" y="380" width="640" height="20" fill="#7f8c8d"/>
        <!-- Vehicles with detection boxes -->
        <!-- Car 1 - Red -->
        <rect x="80" y="165" width="75" height="45" fill="#e74c3c" rx="5"/>
        <rect x="85" y="170" width="25" height="15" fill="rgba(100,150,200,0.8)"/>
        <circle cx="95" cy="210" r="8" fill="#1a1a1a"/><circle cx="145" cy="210" r="8" fill="#1a1a1a"/>
        <rect x="75" y="160" width="85" height="55" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="90" y="155" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 94.2%</text>
        <!-- Car 2 - Blue -->
        <rect x="220" y="168" width="70" height="42" fill="#3498db" rx="5"/>
        <rect x="225" y="173" width="22" height="14" fill="rgba(100,150,200,0.8)"/>
        <circle cx="235" cy="210" r="8" fill="#1a1a1a"/><circle cx="275" cy="210" r="8" fill="#1a1a1a"/>
        <rect x="215" y="163" width="80" height="52" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="225" y="158" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 91.5%</text>
        <!-- Truck -->
        <rect x="380" y="155" width="100" height="60" fill="#f39c12"/>
        <rect x="380" y="160" width="70" height="50" fill="#d68910"/>
        <rect x="455" y="165" width="25" height="20" fill="rgba(100,150,200,0.8)"/>
        <circle cx="400" cy="215" r="10" fill="#1a1a1a"/><circle cx="470" cy="215" r="10" fill="#1a1a1a"/>
        <circle cx="430" cy="215" r="10" fill="#1a1a1a"/>
        <rect x="375" y="150" width="110" height="70" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="385" y="145" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Truck 96.8%</text>
        <!-- Car 3 - Green -->
        <rect x="520" y="168" width="68" height="42" fill="#27ae60" rx="5"/>
        <rect x="525" y="173" width="22" height="14" fill="rgba(100,150,200,0.8)"/>
        <circle cx="535" cy="210" r="8" fill="#1a1a1a"/><circle cx="573" cy="210" r="8" fill="#1a1a1a"/>
        <rect x="515" y="163" width="78" height="52" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="525" y="158" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 92.1%</text>
        <!-- Bottom lane -->
        <!-- Bus -->
        <rect x="120" y="275" width="130" height="55" fill="#e74c3c" rx="3"/>
        <g fill="rgba(100,150,200,0.8)">
            <rect x="125" y="280" width="18" height="20"/><rect x="148" y="280" width="18" height="20"/>
            <rect x="171" y="280" width="18" height="20"/><rect x="194" y="280" width="18" height="20"/>
        </g>
        <circle cx="140" cy="330" r="10" fill="#1a1a1a"/><circle cx="230" cy="330" r="10" fill="#1a1a1a"/>
        <rect x="115" y="270" width="140" height="65" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="125" y="265" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Bus 97.3%</text>
        <!-- Truck 2 -->
        <rect x="400" y="278" width="95" height="55" fill="#7f8c8d"/>
        <rect x="400" y="283" width="65" height="45" fill="#636e72"/>
        <rect x="468" y="288" width="22" height="18" fill="rgba(100,150,200,0.8)"/>
        <circle cx="420" cy="333" r="10" fill="#1a1a1a"/><circle cx="475" cy="333" r="10" fill="#1a1a1a"/>
        <rect x="395" y="273" width="105" height="65" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="405" y="268" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Truck 95.6%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_night_traffic():
    """Night traffic scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <defs>
            <linearGradient id="nightSky" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#0a0a1a"/>
                <stop offset="100%" style="stop-color:#1a1a2e"/>
            </linearGradient>
        </defs>
        <!-- Night sky -->
        <rect width="640" height="150" fill="url(#nightSky)"/>
        <!-- Stars -->
        <g fill="#fff">
            <circle cx="50" cy="30" r="1"/><circle cx="120" cy="60" r="1.5"/>
            <circle cx="200" cy="25" r="1"/><circle cx="280" cy="70" r="1"/>
            <circle cx="350" cy="40" r="1.5"/><circle cx="420" cy="55" r="1"/>
            <circle cx="500" cy="35" r="1"/><circle cx="580" cy="65" r="1.5"/>
            <circle cx="150" cy="90" r="1"/><circle cx="300" cy="100" r="1"/>
            <circle cx="450" cy="85" r="1"/><circle cx="550" cy="95" r="1"/>
        </g>
        <!-- Moon -->
        <circle cx="100" cy="50" r="25" fill="#f5f5f5"/>
        <!-- Night buildings -->
        <rect x="30" y="50" width="80" height="100" fill="#0a0a15"/>
        <rect x="130" y="70" width="60" height="80" fill="#0f0f1a"/>
        <rect x="210" y="40" width="90" height="110" fill="#0a0a15"/>
        <rect x="320" y="60" width="70" height="90" fill="#12121f"/>
        <rect x="410" y="30" width="100" height="120" fill="#0a0a15"/>
        <rect x="530" y="50" width="80" height="100" fill="#0f0f1a"/>
        <!-- Building lights -->
        <g fill="#f4d03f" opacity="0.8">
            <rect x="40" y="60" width="12" height="12"/><rect x="65" y="85" width="12" height="12"/>
            <rect x="140" y="80" width="12" height="12"/><rect x="165" y="105" width="12" height="12"/>
            <rect x="220" y="50" width="12" height="12"/><rect x="245" y="75" width="12" height="12"/>
            <rect x="420" y="40" width="12" height="12"/><rect x="445" y="65" width="12" height="12"/>
            <rect x="470" y="90" width="12" height="12"/><rect x="540" y="60" width="12" height="12"/>
        </g>
        <!-- Road -->
        <rect x="0" y="150" width="640" height="250" fill="#1a1a2e"/>
        <!-- Lane markings -->
        <line x1="0" y1="200" x2="640" y2="200" stroke="#555" stroke-width="3" stroke-dasharray="40,20"/>
        <line x1="0" y1="250" x2="640" y2="250" stroke="#444" stroke-width="2" stroke-dasharray="20,10"/>
        <!-- Vehicles with headlights -->
        <!-- Car 1 -->
        <rect x="80" y="165" width="70" height="42" fill="#1a1a1a" rx="3"/>
        <rect x="85" y="170" width="22" height="14" fill="rgba(100,150,200,0.5)"/>
        <circle cx="95" cy="207" r="8" fill="#1a1a1a"/>
        <circle cx="140" cy="207" r="8" fill="#1a1a1a"/>
        <rect x="78" y="162" width="78" height="50" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="88" y="157" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 88.5%</text>
        <!-- Car 2 with headlights -->
        <rect x="250" y="168" width="68" height="40" fill="#2c3e50" rx="3"/>
        <rect x="255" y="173" width="20" height="13" fill="rgba(100,150,200,0.5)"/>
        <circle cx="265" cy="208" r="8" fill="#1a1a1a"/><circle cx="305" cy="208" r="8" fill="#1a1a1a"/>
        <ellipse cx="320" cy="190" rx="25" ry="15" fill="rgba(255,255,200,0.15)"/>
        <rect x="248" y="165" width="75" height="48" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="258" y="160" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 86.2%</text>
        <!-- Truck -->
        <rect x="420" y="155" width="100" height="58" fill="#34495e"/>
        <rect x="420" y="160" width="70" height="48" fill="#2c3e50"/>
        <rect x="492" y="165" width="24" height="20" fill="rgba(100,150,200,0.5)"/>
        <circle cx="440" cy="213" r="10" fill="#1a1a1a"/><circle cx="500" cy="213" r="10" fill="#1a1a1a"/>
        <circle cx="470" cy="213" r="10" fill="#1a1a1a"/>
        <rect x="415" y="150" width="110" height="68" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="425" y="145" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Truck 94.1%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_highway():
    """Highway scene with fast traffic"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <defs>
            <linearGradient id="highwaySky" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#5dade2"/>
                <stop offset="100%" style="stop-color:#85c1e9"/>
            </linearGradient>
        </defs>
        <!-- Sky -->
        <rect width="640" height="120" fill="url(#highwaySky)"/>
        <!-- Clouds -->
        <g fill="#fff" opacity="0.8">
            <ellipse cx="100" cy="50" rx="50" ry="25"/>
            <ellipse cx="130" cy="45" rx="35" ry="20"/>
            <ellipse cx="400" cy="70" rx="45" ry="22"/>
            <ellipse cx="430" cy="65" rx="30" ry="18"/>
            <ellipse cx="550" cy="40" rx="40" ry="20"/>
        </g>
        <!-- Mountains -->
        <polygon points="0,120 100,60 200,120" fill="#7d9cb5"/>
        <polygon points="150,120 280,40 410,120" fill="#6b8ca4"/>
        <polygon points="350,120 480,50 610,120" fill="#7d9cb5"/>
        <!-- Highway -->
        <rect x="0" y="120" width="640" height="280" fill="#4a4a4a"/>
        <!-- Lane dividers -->
        <line x1="0" y1="180" x2="640" y2="180" stroke="#fff" stroke-width="3" stroke-dasharray="50,30"/>
        <line x1="0" y1="240" x2="640" y2="240" stroke="#fff" stroke-width="3" stroke-dasharray="50,30"/>
        <line x1="0" y1="300" x2="640" y2="300" stroke="#fff" stroke-width="3" stroke-dasharray="50,30"/>
        <!-- Speed vehicle -->
        <rect x="80" y="160" width="90" height="50" fill="#e74c3c" rx="8"/>
        <rect x="85" y="165" width="30" height="18" fill="rgba(100,150,200,0.8)"/>
        <rect x="120" y="165" width="25" height="18" fill="rgba(100,150,200,0.8)"/>
        <circle cx="100" cy="210" r="10" fill="#1a1a1a"/><circle cx="155" cy="210" r="10" fill="#1a1a1a"/>
        <rect x="75" y="155" width="100" height="60" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="85" y="150" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 95.8%</text>
        <!-- Fast car -->
        <rect x="320" y="165" width="85" height="48" fill="#3498db" rx="6"/>
        <rect x="325" y="170" width="28" height="16" fill="rgba(100,150,200,0.8)"/>
        <circle cx="340" cy="213" r="9" fill="#1a1a1a"/><circle cx="390" cy="213" r="9" fill="#1a1a1a"/>
        <rect x="315" y="160" width="95" height="58" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="325" y="155" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Car 93.4%</text>
        <!-- Truck -->
        <rect x="500" y="155" width="110" height="62" fill="#95a5a6"/>
        <rect x="500" y="160" width="75" height="52" fill="#7f8c8d"/>
        <rect x="578" y="165" width="28" height="22" fill="rgba(100,150,200,0.8)"/>
        <circle cx="520" cy="217" r="11" fill="#1a1a1a"/><circle cx="595" cy="217" r="11" fill="#1a1a1a"/>
        <circle cx="560" cy="217" r="11" fill="#1a1a1a"/>
        <rect x="495" y="150" width="120" height="72" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="505" y="145" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Truck 96.2%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_bus_scene():
    """Bus detection scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <rect width="640" height="400" fill="#87CEEB"/>
        <!-- Buildings -->
        <rect x="20" y="80" width="100" height="120" fill="#34495e"/>
        <rect x="140" y="60" width="80" height="140" fill="#3d566e"/>
        <rect x="240" y="100" width="90" height="100" fill="#2c3e50"/>
        <rect x="450" y="70" width="120" height="130" fill="#34495e"/>
        <rect x="520" y="50" width="100" height="150" fill="#3d566e"/>
        <!-- Sidewalk -->
        <rect x="0" y="250" width="640" height="50" fill="#95a5a6"/>
        <rect x="0" y="300" width="640" height="100" fill="#4a4a4a"/>
        <!-- Lane marking -->
        <line x1="0" y1="340" x2="640" y2="340" stroke="#fff" stroke-width="2" stroke-dasharray="30,20"/>
        <!-- Big Bus -->
        <rect x="200" y="160" width="200" height="100" fill="#f1c40f" rx="5"/>
        <g fill="#2c3e50">
            <rect x="210" y="170" width="25" height="35" rx="3"/>
            <rect x="245" y="170" width="25" height="35" rx="3"/>
            <rect x="280" y="170" width="25" height="35" rx="3"/>
            <rect x="315" y="170" width="25" height="35" rx="3"/>
            <rect x="350" y="170" width="25" height="35" rx="3"/>
            <rect x="385" y="170" width="25" height="35" rx="3"/>
        </g>
        <circle cx="240" cy="260" r="18" fill="#1a1a1a"/>
        <circle cx="370" cy="260" r="18" fill="#1a1a1a"/>
        <!-- Bus detection box -->
        <rect x="195" y="155" width="210" height="115" fill="none" stroke="#00ff00" stroke-width="4"/>
        <text x="205" y="145" fill="#00ff00" font-family="Arial" font-size="14" font-weight="bold">Bus 98.5%</text>
        <!-- Small car -->
        <rect x="80" y="280" width="70" height="45" fill="#e74c3c" rx="5"/>
        <rect x="85" y="285" width="22" height="15" fill="rgba(100,150,200,0.8)"/>
        <circle cx="95" cy="325" r="8" fill="#1a1a1a"/><circle cx="135" cy="325" r="8" fill="#1a1a1a"/>
        <rect x="75" y="275" width="80" height="55" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="85" y="270" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Car 91.2%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_truck_scene():
    """Truck detection scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <rect width="640" height="400" fill="#aed6f1"/>
        <!-- Mountains -->
        <polygon points="0,150 150,50 300,150" fill="#85929e"/>
        <polygon points="200,150 400,30 600,150" fill="#7f8c8d"/>
        <!-- Road -->
        <rect x="0" y="180" width="640" height="220" fill="#4a4a4a"/>
        <!-- Side road markings -->
        <line x1="0" y1="280" x2="640" y2="280" stroke="#fff" stroke-width="3" stroke-dasharray="40,25"/>
        <!-- Industrial building -->
        <rect x="450" y="50" width="180" height="130" fill="#566573"/>
        <rect x="460" y="60" width="40" height="40" fill="#7f8c8d"/>
        <rect x="510" y="60" width="40" height="40" fill="#7f8c8d"/>
        <rect x="560" y="60" width="40" height="40" fill="#7f8c8d"/>
        <!-- Big Truck -->
        <rect x="180" y="140" width="260" height="130" fill="#3498db"/>
        <rect x="180" y="150" width="180" height="115" fill="#2980b9"/>
        <rect x="365" y="155" width="70" height="50" fill="#2c3e50"/>
        <rect x="375" y="162" width="50" height="35" fill="rgba(100,150,200,0.8)"/>
        <circle cx="210" cy="270" r="18" fill="#1a1a1a"/>
        <circle cx="280" cy="270" r="18" fill="#1a1a1a"/>
        <circle cx="350" cy="270" r="18" fill="#1a1a1a"/>
        <circle cx="420" cy="270" r="18" fill="#1a1a1a"/>
        <rect x="175" y="135" width="270" height="145" fill="none" stroke="#00ff00" stroke-width="4"/>
        <text x="185" y="125" fill="#00ff00" font-family="Arial" font-size="14" font-weight="bold">Truck 97.8%</text>
        <!-- Small car -->
        <rect x="80" y="290" width="75" height="48" fill="#27ae60" rx="5"/>
        <rect x="85" y="295" width="25" height="16" fill="rgba(100,150,200,0.8)"/>
        <circle cx="95" cy="338" r="9" fill="#1a1a1a"/><circle cx="140" cy="338" r="9" fill="#1a1a1a"/>
        <rect x="75" y="285" width="85" height="58" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="85" y="280" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Car 92.3%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_sport_car():
    """Sport car scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <rect width="640" height="400" fill="#2ecc71"/>
        <!-- Mountains -->
        <polygon points="0,200 100,100 200,200" fill="#27ae60"/>
        <polygon points="150,200 300,50 450,200" fill="#229954"/>
        <polygon points="400,200 500,80 600,200" fill="#27ae60"/>
        <!-- Road with blur effect for speed -->
        <rect x="0" y="220" width="640" height="180" fill="#333"/>
        <line x1="0" y1="300" x2="640" y2="300" stroke="#fff" stroke-width="4" stroke-dasharray="60,40"/>
        <!-- Motion lines -->
        <g stroke="#fff" stroke-width="2" opacity="0.3">
            <line x1="0" y1="250" x2="100" y2="250"/>
            <line x1="20" y1="270" x2="80" y2="270"/>
            <line x1="0" y1="350" x2="120" y2="350"/>
            <line x1="30" y1="370" x2="90" y2="370"/>
        </g>
        <!-- Sport car - side view moving fast -->
        <rect x="250" y="240" width="160" height="55" fill="#e74c3c" rx="8"/>
        <path d="M270 240 Q320 200 380 200 L400 240 Z" fill="#c0392b"/>
        <rect x="280" y="210" width="80" height="30" fill="rgba(100,150,200,0.8)"/>
        <circle cx="280" cy="295" r="15" fill="#1a1a1a"/><circle cx="390" cy="295" r="15" fill="#1a1a1a"/>
        <circle cx="280" cy="295" r="8" fill="#666"/><circle cx="390" cy="295" r="8" fill="#666"/>
        <!-- Speed lines behind car -->
        <g stroke="#e74c3c" stroke-width="3" opacity="0.6">
            <line x1="230" y1="250" x2="180" y2="250"/>
            <line x1="240" y1="270" x2="150" y2="270"/>
            <line x1="235" y1="285" x2="160" y2="285"/>
        </g>
        <!-- Detection box -->
        <rect x="245" y="235" width="170" height="70" fill="none" stroke="#00ff00" stroke-width="4"/>
        <text x="255" y="225" fill="#00ff00" font-family="Arial" font-size="14" font-weight="bold">Car 98.2%</text>
        <!-- Background cars -->
        <rect x="50" y="260" width="60" height="35" fill="#3498db" rx="3"/>
        <circle cx="60" cy="295" r="7" fill="#1a1a1a"/><circle cx="95" cy="295" r="7" fill="#1a1a1a"/>
        <rect x="45" y="255" width="70" height="45" fill="none" stroke="#00ff00" stroke-width="2"/>
        <text x="52" y="250" fill="#00ff00" font-family="Arial" font-size="10">Car 89.5%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_street():
    """Urban street scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <rect width="640" height="400" fill="#87CEEB"/>
        <!-- Street view -->
        <rect x="0" y="150" width="640" height="250" fill="#4a4a4a"/>
        <!-- Sidewalk -->
        <rect x="0" y="130" width="640" height="30" fill="#95a5a6"/>
        <!-- Buildings -->
        <rect x="20" y="30" width="80" height="100" fill="#8e6f4a"/>
        <rect x="120" y="50" width="100" height="80" fill="#7d6b5a"/>
        <rect x="240" y="20" width="70" height="110" fill="#8e6f4a"/>
        <rect x="330" y="40" width="90" height="90" fill="#7d6b5a"/>
        <rect x="440" y="25" width="80" height="105" fill="#8e6f4a"/>
        <rect x="540" y="55" width="90" height="75" fill="#7d6b5a"/>
        <!-- Shop signs -->
        <rect x="130" y="60" width="80" height="20" fill="#e74c3c"/>
        <text x="140" y="75" fill="#fff" font-family="Arial" font-size="12">商店</text>
        <rect x="350" y="50" width="70" height="18" fill="#3498db"/>
        <text x="358" y="64" fill="#fff" font-family="Arial" font-size="11">餐厅</text>
        <!-- Trees -->
        <circle cx="100" cy="120" r="20" fill="#27ae60"/>
        <rect x="97" y="130" width="6" height="20" fill="#8B4513"/>
        <circle cx="300" cy="115" r="22" fill="#2ecc71"/>
        <rect x="297" y="125" width="6" height="18" fill="#8B4513"/>
        <circle cx="500" cy="118" r="18" fill="#27ae60"/>
        <rect x="497" y="128" width="6" height="17" fill="#8B4513"/>
        <!-- Lane marking -->
        <line x1="0" y1="220" x2="640" y2="220" stroke="#fff" stroke-width="2" stroke-dasharray="25,15"/>
        <line x1="0" y1="300" x2="640" y2="300" stroke="#fff" stroke-width="2" stroke-dasharray="25,15"/>
        <!-- Vehicles -->
        <rect x="80" y="170" width="75" height="48" fill="#9b59b6" rx="5"/>
        <rect x="85" y="175" width="24" height="16" fill="rgba(100,150,200,0.8)"/>
        <circle cx="95" cy="218" r="9" fill="#1a1a1a"/><circle cx="140" cy="218" r="9" fill="#1a1a1a"/>
        <rect x="75" y="165" width="85" height="58" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="85" y="160" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Car 90.8%</text>
        <rect x="280" y="172" width="70" height="45" fill="#1abc9c" rx="5"/>
        <rect x="285" y="177" width="22" height="15" fill="rgba(100,150,200,0.8)"/>
        <circle cx="295" cy="217" r="8" fill="#1a1a1a"/><circle cx="335" cy="217" r="8" fill="#1a1a1a"/>
        <rect x="275" y="167" width="80" height="55" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="285" y="162" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Car 88.3%</text>
        <rect x="450" y="250" width="90" height="55" fill="#34495e"/>
        <rect x="450" y="255" width="60" height="45" fill="#2c3e50"/>
        <rect x="512" y="260" width="24" height="20" fill="rgba(100,150,200,0.8)"/>
        <circle cx="465" cy="305" r="10" fill="#1a1a1a"/><circle cx="525" cy="305" r="10" fill="#1a1a1a"/>
        <rect x="445" y="245" width="100" height="65" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="455" y="240" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Truck 93.7%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def generate_motorway():
    """Motorway toll scene"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">
        <rect width="640" height="400" fill="#5dade2"/>
        <!-- Clouds -->
        <g fill="#fff" opacity="0.7">
            <ellipse cx="80" cy="40" rx="40" ry="20"/>
            <ellipse cx="300" cy="60" rx="50" ry="25"/>
            <ellipse cx="500" cy="35" rx="45" ry="22"/>
        </g>
        <!-- Motorway -->
        <rect x="0" y="160" width="640" height="240" fill="#4a4a4a"/>
        <!-- Toll booth area -->
        <rect x="0" y="160" width="150" height="240" fill="#5d6d7e"/>
        <rect x="20" y="200" width="50" height="40" fill="#e74c3c"/>
        <rect x="80" y="200" width="50" height="40" fill="#27ae60"/>
        <!-- Lane markings -->
        <line x1="150" y1="200" x2="640" y2="200" stroke="#fff" stroke-width="4" stroke-dasharray="50,30"/>
        <line x1="150" y1="280" x2="640" y2="280" stroke="#fff" stroke-width="4" stroke-dasharray="50,30"/>
        <!-- Bus -->
        <rect x="180" y="165" width="150" height="85" fill="#f1c40f" rx="3"/>
        <g fill="#2c3e50">
            <rect x="188" y="175" width="22" height="30" rx="2"/>
            <rect x="215" y="175" width="22" height="30" rx="2"/>
            <rect x="242" y="175" width="22" height="30" rx="2"/>
            <rect x="269" y="175" width="22" height="30" rx="2"/>
            <rect x="296" y="175" width="22" height="30" rx="2"/>
        </g>
        <circle cx="210" cy="250" r="14" fill="#1a1a1a"/>
        <circle cx="310" cy="250" r="14" fill="#1a1a1a"/>
        <rect x="175" y="160" width="160" height="95" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="185" y="152" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">Bus 96.8%</text>
        <!-- Car -->
        <rect x="400" y="172" width="70" height="45" fill="#3498db" rx="5"/>
        <rect x="405" y="177" width="23" height="15" fill="rgba(100,150,200,0.8)"/>
        <circle cx="415" cy="217" r="8" fill="#1a1a1a"/><circle cx="455" cy="217" r="8" fill="#1a1a1a"/>
        <rect x="395" y="167" width="80" height="55" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="405" y="162" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Car 91.4%</text>
        <!-- Truck -->
        <rect x="520" y="240" width="100" height="60" fill="#7f8c8d"/>
        <rect x="520" y="245" width="68" height="50" fill="#636e72"/>
        <rect x="590" y="250" width="26" height="22" fill="rgba(100,150,200,0.8)"/>
        <circle cx="538" cy="300" r="11" fill="#1a1a1a"/><circle cx="605" cy="300" r="11" fill="#1a1a1a"/>
        <circle cx="572" cy="300" r="11" fill="#1a1a1a"/>
        <rect x="515" y="235" width="110" height="70" fill="none" stroke="#00ff00" stroke-width="3"/>
        <text x="525" y="230" fill="#00ff00" font-family="Arial" font-size="11" font-weight="bold">Truck 95.1%</text>
    </svg>'''
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

def main():
    # Generate all images
    images = {
        'city_road': generate_city_road(),
        'night_traffic': generate_night_traffic(),
        'highway': generate_highway(),
        'motorway': generate_motorway(),
        'street': generate_street(),
        'bus': generate_bus_scene(),
        'truck': generate_truck_scene(),
        'sport_car': generate_sport_car(),
    }
    
    # Output JavaScript file
    output = '// Auto-generated vehicle detection images\nconst BASE64_IMAGES = ' + json.dumps(images, indent=2) + ';\n\n'
    output += '''
// Auto-initialize when DOM is ready
if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", function() {
        for (const [name, src] of Object.entries(BASE64_IMAGES)) {
            const img = document.getElementById("gallery-" + name);
            if (img && src) {
                img.src = src;
                img.onload = function() { this.classList.add("loaded"); };
                img.onerror = function() { this.classList.add("loaded"); };
            }
        }
    });
}
'''
    
    with open('images_base64.js', 'w') as f:
        f.write(output)
    
    print(f"Generated {len(images)} images")
    print("File: images_base64.js")

if __name__ == '__main__':
    main()
