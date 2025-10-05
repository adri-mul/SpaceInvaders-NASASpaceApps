import math
import csv
import json

# Calculate the radius of a star using the Stefan-Boltzmann law.
def calculate_star_radius(luminosity, temperature):
    stefan_boltzmann_constant = 5.670374419e-8  # W/m^2/K^4
    radius = math.sqrt(luminosity / (4 * math.pi * stefan_boltzmann_constant * temperature**4))
    return radius

# Calculate the inner and outer boundaries of the habitable zone in AU.
def calculate_habitable_zone(luminosity):
    inner_boundary = math.sqrt(luminosity / 1.1)
    outer_boundary = math.sqrt(luminosity / 0.53)
    return inner_boundary, outer_boundary

# Calculate the orbital distance of a planet given the stellar flux and luminosity.
# Only valid for orbits with low eccentricity.
def calculate_orbital_distance(flux, luminosity):
    distance = math.sqrt(luminosity / (4 * math.pi * flux))
    return distance

# Calculate the mass of a star using the mass-luminosity relation.
# Only valid for main-sequence stars.
def calculate_stellar_mass(luminosity):
    sun_mass = 1.989e30  # kg
    sun_luminosity = 3.828e26  # W
    return math.pow(luminosity / sun_luminosity, 1/3.5) * sun_mass

# Calculate the orbital distance of a planet using Kepler's third law.
def calculate_orbital_distance(orbital_period, stellar_mass):
    gravitational_constant = 6.67430e-11  # m^3/kg/s^2
    period = math.cbrt((orbital_period**2 * gravitational_constant * stellar_mass) / (4 * math.pi**2))
    return period

def calculate_planets_in_habitable_zone(planets):
    """
    Given a star's luminosity (in Solar units) and an iterable of planet rows (as
    returned by csv.DictReader), return a list
    of planet rows that fall inside the habitable zone.

    This function is robust to CSV rows that provide:
    - pl_insol (planet insolation in Earth flux units) -> distance (AU) = sqrt(L / S)
    - pl_orbper (orbital period in days) -> use Kepler's third law and the star mass
    - an explicit orbital distance field (common names are checked)

    The returned planet rows are shallow-copies of the input rows with an added
    key 'computed_orbital_distance_au' when the distance had to be derived.
    """

    

    # Constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    AU = 1.495978707e11  # m
    sun_mass = 1.989e30  # kg

    def _to_float(value):
        try:
            if value is None or value == '':
                return None
            return float(value)
        except Exception:
            return None

    habitable_planets = []

    for planet in planets:
        # For Kepler CSV: koi_period, koi_srad, koi_steff, koi_insol, koi_prad, etc.
        period_days = _to_float(planet.get('koi_period'))
        star_rad_rsun = _to_float(planet.get('koi_srad'))
        teff_k = _to_float(planet.get('koi_steff'))
        insolation = _to_float(planet.get('koi_insol'))
        # Mass is not directly available, but can be estimated from radius if needed
        luminosity = None

        # Estimate luminosity if possible (L = 4 * pi * R^2 * sigma * T^4)
        if star_rad_rsun is not None and teff_k is not None:
            R = star_rad_rsun * 6.957e8
            sigma = 5.670374419e-8
            luminosity = 4 * math.pi * R**2 * sigma * teff_k**4
            luminosity /= 3.828e26

        # If not, skip planet
        if luminosity is None:
            continue

        inner_hz, outer_hz = calculate_habitable_zone(luminosity)

        # Determine orbital distance
        distance_au = None
        # Prefer insolation if available
        if insolation is not None and luminosity is not None and insolation > 0:
            try:
                distance_au = math.sqrt(luminosity / insolation)
            except Exception:
                distance_au = None
        # Otherwise, use period and estimate mass from radius (roughly)
        elif period_days is not None and star_rad_rsun is not None:
            # Estimate mass from radius: M ~ R^0.8 (rough, for main sequence)
            star_mass_msun = star_rad_rsun ** 0.8
            P = period_days * 86400.0
            M = star_mass_msun * sun_mass
            try:
                a_m = ((G * M * P**2) / (4 * math.pi**2)) ** (1/3)
                distance_au = a_m / AU
            except Exception:
                distance_au = None

        if distance_au is None:
            continue

        if inner_hz <= distance_au <= outer_hz:
            row_copy = dict(planet)
            row_copy['computed_orbital_distance_au'] = distance_au
            row_copy['computed_luminosity_solar'] = luminosity
            habitable_planets.append(row_copy)

    # Return as JSON object
    return json.dumps(habitable_planets, indent=2)

def test_script():
    FILE = "Data/cumulative_2025.10.04_09.06.58.csv"
    with open(FILE) as csvReader:
        filtered_lines = (line for line in csvReader if not line.lstrip().startswith('#'))
        reader = csv.DictReader(filtered_lines)
        return calculate_planets_in_habitable_zone(reader)

demo = test_script()
print(len(json.loads(demo)))
