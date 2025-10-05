import math

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

def calculate_planets_in_habitable_zone(star_luminosity, planets):
    """
    Given a star's luminosity (in Solar units) and an iterable of planet rows (as
    returned by csv.DictReader or a list of mapping-like objects), return a list
    of planet rows that fall inside the habitable zone.

    This function is robust to CSV rows that provide:
    - pl_insol (planet insolation in Earth flux units) -> distance (AU) = sqrt(L / S)
    - pl_orbper (orbital period in days) -> use Kepler's third law and the star mass
    - an explicit orbital distance field (common names are checked)

    The returned planet rows are shallow-copies of the input rows with an added
    key 'computed_orbital_distance_au' when the distance had to be derived.
    """

    inner_hz, outer_hz = calculate_habitable_zone(star_luminosity)
    habitable_planets = []

    # physical constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    AU = 1.495978707e11  # m

    # precompute stellar mass.
    # Note: many CSV catalogs (like the TOI file) give stellar luminosity in
    # solar units (L_sun = 1). The helper `calculate_stellar_mass` above
    # expects luminosity in Watts (it divides by sun_luminosity = 3.828e26).
    # To be robust we handle both cases:
    # - if star_luminosity is small (e.g. < 1e5) treat it as solar units
    #   and use the mass-luminosity relation: M = (L/L_sun)^(1/3.5) * M_sun
    # - otherwise, assume it's already in Watts and call the helper.
    sun_mass = 1.989e30  # kg
    sun_luminosity_w = 3.828e26  # W
    try:
        if star_luminosity is None:
            stellar_mass = sun_mass
        elif float(star_luminosity) < 1e5:
            stellar_mass = (float(star_luminosity) ** (1/3.5)) * sun_mass
        else:
            stellar_mass = calculate_stellar_mass(float(star_luminosity))
    except Exception:
        stellar_mass = sun_mass

    def _to_float(value):
        try:
            if value is None or value == '':
                return None
            return float(value)
        except Exception:
            return None

    # helper to fetch from dict-like planet rows
    def _pick_field(row, candidates):
        for k in candidates:
            if k in row and row[k] not in (None, ''):
                return row[k]
        return None

    for planet in planets:
        # Expect CSV row as a mapping (e.g., csv.DictReader). If it's not a mapping
        # skip it — caller can transform rows to dicts before calling.
        if not hasattr(planet, 'get') and not isinstance(planet, dict):
            # not a dict-like row; skip
            continue

        # Try explicit orbital distance first (common names)
        dist_val = _pick_field(planet, [
            'orbital_distance', 'pl_orbsma', 'pl_a', 'a', 'semi_major_axis'
        ])
        distance_au = _to_float(dist_val) if dist_val is not None else None

        # If we don't have distance, try insolation (pl_insol) which is in Earth flux.
        if distance_au is None:
            insol_val = _pick_field(planet, ['pl_insol', 'insolation', 'stellar_flux', 'st_flux'])
            insol = _to_float(insol_val)
            if insol is not None and insol > 0:
                # For insolation in Earth flux and luminosity in Solar units:
                # distance (AU) = sqrt(L / S)
                try:
                    distance_au = math.sqrt(star_luminosity / insol)
                except Exception:
                    distance_au = None

        # If still no distance, try orbital period + Kepler's 3rd law
        if distance_au is None:
            per_val = _pick_field(planet, ['pl_orbper', 'orbital_period', 'period_days'])
            per_days = _to_float(per_val)
            if per_days is not None and per_days > 0:
                # Convert days to seconds
                P = per_days * 86400.0
                try:
                    a_m = ((G * stellar_mass * P**2) / (4 * math.pi**2)) ** (1/3)
                    distance_au = a_m / AU
                except Exception:
                    distance_au = None

        if distance_au is None:
            # Not enough information for this planet
            continue

        # Check habitable zone (inner_hz and outer_hz are in AU)
        if inner_hz <= distance_au <= outer_hz:
            # Return a shallow copy and annotate the computed distance for clarity
            row_copy = dict(planet)
            row_copy['computed_orbital_distance_au'] = distance_au
            habitable_planets.append(row_copy)

    return habitable_planets