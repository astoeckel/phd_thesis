import numpy as np
import json
import re
import sys
from unidecode import unidecode

# Manually excluded false-positive matches
blacklist = {
    "13b69333d2828f4fb93ca514482fdc39b5717049",
    "d7e98345f247d421c7635a77a03e52bd6311bab9",
    "1aa1f4ff29d1f24175d66af8dd72919c0420c1ad",
    "755b2e1d4cb7eb944568d64cfb8b44015ab0c30b",
    "3923d52930b9bd6f51e375967824762bed1e6487",
    "fca02804a905f72f3cc505b55afd63a4c239f3af",
    "c554d1d4c3908678505bba4c2588071430b385d1",
    "9be3f689cc0ff6dadd88ca36c95fbe879de39f07",
    "21ffa83ccc94bc3ee0432faa5554ec2e836c6680",
    "773c5afb42ef09db49ab84a8e3230b739edcc70a",
    "1c0da4e7ac02f2985e9a5f0c424ab8f98398917e",
    "fe8b12e87821ee3d1ab9d8c56c4eb66672fc4305",
}

# Manual categorisation
kinds = {
    "32bec6de3a752c5da22e55c24253dc08a35aade2": "theory",
    "50719ba496dd549884acf70c83e331585e129d30": "theory",
    "077f5545c896eefaa551e2848f8a762114b6a77e": "biology",
    "55e4f060e8e27c2383a683da277b9e5c8cb6e922": "theory",
    "4aaa95b6027ceb3812547dd17ab6245e80edd8c4": "other",
    "71ba26ae203a056f15c76575bd0befae54304cbd": "biology",
    "068597d57d273f1d1429ab9467c3b4ab70338b06": "theory",
    "255021bcfd296479f2ddd22a5561446b5206a9dc": "other",
    "6e310bd014fc002c7f589adac9d6edc69472114d": "biology",
    "0c7cb6e603dcc59fe2a031caeb57bc1c862e4624": "software",
    "b5bc53c5339e391611cfabe602a005eb12df7b1b": "biology",
    "4c521fa67ab54a3e8041aed68bdb8d25a83336ba": "theory",
    "ad09e494a9b789f75256c58b4f5979cb7168e178": "theory",
    "42a816e3590bc9961cabe311e1b95c412a6a13f3": "biology",
    "b8d3d178273b0a2d1cf5f78a2957139f6c244e7f": "other",
    "d64cf08cd925b8b40f5d9b66e3a328fc12ea7d48": "other",
    "bae6c7e8a46e26d13e5db1ca2976d73c41725b63": "biology",
    "1212eaa8317776c59fb5ec5ef0ff0653082a41a0": "other",
    "2208f21dd7aed3bef69a5455b47fc1ab160568b3": "neuromorphics",
    "a053a0ebd7ddc90254f07aa798e73161bf3b3edd": "other",
    "3214a867388b3e275b4581d9ca894fbf7b7ab06d": "biology",
    "2df578bf6ea80e676c1a98e3966879270685f3ce": "theory",
    "a681a06936105321b7f07a5446e69e899a766e9d": "other",
    "a5c419fcd6ea6f33be067b665e77e3518853a2d1": "biology",
    "174d050ace5b9dcbd637b42eb00a349e37e37434": "neuromorphics",
    "908629e39bc6b6e287118dec7e656d2395ad1382": "neuromorphics",
    "b0b90246982b2b3704abfb1862823f5a15de0988": "other",
    "556b7915d2e70fd2e6dc563a635b96faa239635a": "biology",
    "cd72c5de72c581e28c27d50660728e3d38de395a": "theory",
    "a1b536373b36a651da64a6d42713dcafef2ea071": "theory",
    "13cc0d464ae960d67d6bef22ddf07dfdbb47dfca": "neuromorphics",
    "9dd32f3a0d524ac16d522f8745c67a906dcdaac6": "other",
    "b0e95d4e32c57dfe2001482203b676f272a74417": "theory",
    "f47aa2a558a10379f776357021d0b576cdfb92b5": "biology",
    "b3900904f0098cf9af135c8c13cfdea5a5904af7": "theory",
    "94b4c03c5d2661fa232e14eca1758c3d9c5cd1b4": "biology",
    "329f242e21b0c87e7fbfa41a57fa1413e729332a": "neuromorphics",
    "9664ab17170e0442f35e27d8b3fac7398aa12d08": "other",
    "67d18898e8453029ab732ebf0460459c7462209b": "neuromorphics",
    "56492d19d384b6099824c3d1cd6819071c0d4de8": "neuromorphics",
    "ea37eca73eee1c12ced43abb563c7bc7773fb7ae": "theory",
    "21892c4a7bef2b90cd2c34a2bdac63effe379b8d": "neuromorphics",
    "c8f233be4995141c3f9aa8db52b6f67015ff41d9": "neuromorphics",
    "f5b3769ae7f7a38f8fab4608c0acc33581816bfd": "other",
    "939be55e4c179b8f306445e85656d437282ec1d9": "software",
    "454e2f5eff9d129f738b5ad6c830f63ef1048776": "neuromorphics",
    "483ed39eb4a73adf243e3b42dfc25ee33a15fd7a": "robotics",
    "b5c200ed55d3f7bf536fb9cd0188ae800e7aefd4": "theory",
    "46475abf83907fd91fa8ed3da3fdbc0409ab0328": "biology",
    "0599b345a74342d7a534ba47fc76618ce3f2668b": "biology",
    "b09532dece76bed1b5dbf7eb40da0d2b5e9e93e1": "biology",
    "f893353da02124a97168970c4009975ac74ff071": "biology",
    "d6a4c929628266221e10b7fe4e5badb7d40ce4a4": "biology",
    "5fe00d9024774bd35a8da354d8877e8f7e37acf5": "other",
    "603065cfadc9795867c641c822c575cb89dce35f": "biology",
    "75fe664dca78df6c9012eaaf589120a95faeaa67": "theory",
    "f349eaac26d5b8b7512fbe958a0039cf1b1bb76a": "neuromorphics",
    "499f695e5d4fb82b808d58ec550273b26a5c26e5": "neuromorphics",
    "ab98ca1400462462b413bd34ec57d05e80a971ab": "theory",
    "805bc8321faea55f3fe016ae5ff21ecaf908700c": "neuromorphics",
    "63cdf4da68a91d01f164032efacf63411f9cdd4b": "software",
    "c8624c278849c6a387be4b80476eb9b81ff7003f": "neuromorphics",
    "479ddbf22f3e390161fdbb900f253aea8bc69d3e": "robotics",
    "7ce3e65dd33c0782329bc5d84dce5af8c1f7a05a": "robotics",
    "0297080cae87b284c4cd272f464ef32a18588415": "other",
    "341af63ef4ce4d03fef95cb5d0bce1a3079a962c": "theory",
    "5678278d30d598d044d8e75db421a9d9bc2a0db7": "biology",
    "4c7cc9ea8c87b820ec3df2cda50c34b167604096": "biology",
    "e51956eec3404db7c7c8fdd222c28c576a27cd73": "neuromorphics",
    "8c824c098e5c8bdfb0c3d57561dd0303e252481c": "other",
    "140ff0fcca8899399d9e64823455f0cd0c421f3e": "theory",
    "540880cf42ca4ed39488add16a966a515e9c235b": "biology",
    "d676cc4bff601b0ffa20c20c339b045c998ce1e7": "biology",
    "0d71c9f7a2f5a5671399445347e7cf2067b3e0d6": "biology",
    "60c9bb34a240c65c204d30d5ee1b6736358779c9": "neuromorphics",
    "c78fd64fc222479c4d6d3000f43b4e1496484705": "theory",
    "22462406eb304533e59f7983d3c8896705ae7d4b": "theory",
    "029c0ce91bba1daaa396639f95d50f7a856be6e8": "theory",
    "ba4b683857844492ecc8bff5bc5fcf06d3ec9301": "neuromorphics",
    "c8692f1fefc87eeed371667689c3cd3e36d58354": "other",
    "624684f08a365acd0f1ae5f90ebd16cbb4db27ee": "theory",
    "2bb28e9b04db82627e4ad87233811bdced76d255": "robotics",
    "0178f7275f581e10738c870a8cf005454fd72dcd": "neuromorphics",
    "9b5b5ed906e40f135182e6206dd43910f7043127": "neuromorphics",
    "6168a89efa6c4ec846aa29db996b895d98bea3b0": "neuromorphics",
    "7438c3930044761e24d9c82005085f0d64cea2cd": "biology",
    "0a017b7da2b113194b6937feba3669072bafd626": "neuromorphics",
    "ca2b897f997e2dbce8d9bb5bc6166428ea1c63de": "theory",
    "2dff4bc1918ae7857a158e2f9bf223c07e585fdc": "biology",
    "1ef86385ae33f04d20efbcfff96462a00c1bc8c1": "theory",
    "9e99a83dffd8bb0139aa508da83c22ec19d57bda": "other",
    "59c18488b771434b10ae07951df84ef2d82416e8": "theory",
    "94d24e9c98a720832330107927067a4530a056dc": "neuromorphics",
    "b4af86801d7349d55ec8e8c29543da5832207ab7": "theory",
    "9031403707186c20482276fdb5fffe6cd4d6be3c": "biology",
    "725a23b7fe2c8ea04341c530f3e6428f106db8b9": "biology",
    "3420f24faf20e8a356f63ca88283e9ad96561d15": "software",
    "9c3c4602a2f24a44bb154b750e6a6bdcd5dbbf3f": "neuromorphics",
    "4d71ae29bebaf682cf58741f374879eccb595828": "theory",
    "211f0c79d7843df7bd1d36d280a272c97e357bce": "biology",
    "cb9fb49813f1e86d23bdc2f7df83f4c4774369be": "biology",
    "581aa1385217c0bc3344996ae004a8861a993e2b": "biology",
    "90e02c6febad1fe6b1768a34294938e221b8e379": "biology",
    "86b3f9378d58f0dc8a22250ef3db2432a7bc47c7": "neuromorphics",
    "55ce934130d3bd91f2144fb6efbb239c44822216": "neuromorphics",
    "c445c0f6784817115f2cb2aca78b836ca11113ed": "neuromorphics",
    "61615f51a3a1943a70ccedc64304d2a12deddb17": "other",
    "e1b986fe08c355487b84233d51ffb9a87d97408c": "theory",
    "f31f6728831f6fd50d5d748647d2526469b6d8f6": "neuromorphics",
    "c854390243603045dcf89ed7b33032664c77cd12": "neuromorphics",
    "e4d271024c46b334d28a91a568217f15bd886465": "neuromorphics",
    "38992a122c4be6eab3afe1b394ab7ad238b6133f": "biology",
    "838561581c106875e35d7da896a00a6e3cbc2970": "theory",
    "5dad53635be3759aec29b9cc002a789b78f69064": "software",
    "1660470305f8fdae4bf6246cc8ad5faf242763da": "theory",
    "8b2c7784bbf0d60df910e4389e1a12f1463e281d": "biology",
    "0ee228159153ca7f9e8e3d58832092dbaba1a89b": "other",
    "c06b4c901a5ea091ac4aa604292e0d05bfa058bb": "neuromorphics",
    "5e139d46e054867b9fd262ac0609e426b2941272": "robotics",
    "2d57e9583ddac094e8813a1ba29ecb8249e088af": "software",
    "aff3f14547e8f6f7467eb554c215521b9acee960": "biology",
    "37aeffffe641b0a19ab51f783793c63d7e9a6b1b": "neuromorphics",
    "fbef45e58dfd495ce86ff915c4a9386144e4ccda": "biology",
    "073dbb811b072c5fe2b667e3224b5d592ca6ffbc": "robotics",
    "b31bab0ef7c59f254b246e091034650a67c5055c": "robotics",
    "ec73268b348cccc7d3a0270b2e57ee3a669567ed": "biology",
    "fc53fa6d188785e7a4b22aab349e13cc04b64c19": "neuromorphics",
    "2ccb7d5374cc36f5a423170a6629af12c3c2c18a": "biology",
    "99316df3b069d75f5c59afd5f4b0e88e92ea6360": "robotics",
    "f340f110300b29caaed3fa9e6ff0aa49b2ecb194": "theory",
    "2c1c07240736a46fdc8fac722dce06e03263b6f7": "other",
    "387feeda316870fc10f6dcc8fef54e343bf64530": "biology",
    "5418b8ba379927c04639eba5678ff7bb8c23ebc3": "biology",
    "35e54186ad6a828dd2b0fb5a2066630ed7f5c66b": "theory",
    "542ae508524d951eb4a8d3a3a49c1e0fa4daac28": "neuromorphics",
    "f7ffd705ee87c2b3400893abfcd80891b91f802f": "neuromorphics",
    "167986ee93a5cd1c18b1fb3e1b5b2be6648ee349": "neuromorphics",
    "6f5a4ee6fef1aa8f632efcf695ba394c331f256b": "theory"
}

# List of people in the CNRG according to http://compneuro.uwaterloo.ca/people.html
cnrg_people = [
    "Chris Eliasmith",
    "Terrence C. Stewart",
    "Terrence Steward",  # misspelling
    "Andreas Stöckel",
    "Narsimha R. Chilkuri",
    "Natarajan Vaidyanathan",
    "Nicole Dumont",
    "Pawel Jaworski",
    "Peter Duggins",
    "Thomas Lu",
    "Ivana Kajić",
    "Brent Komer",
    "Aaron R. Voelker",
    "Sverrir Thorgeirsson",
    "Sugandha Sharma",
    "Sean Aubin",
    "Jan Gosmann",
    "Xuan Choo",
    "Ryan Orr",
    "Peter Suma",
    "Eric Hunsberger",
    "Mariah Martin Shein",
    "Peter Blouw",
    "Youssef Zaky",
    "Trevor Bekolay",
    "Travis DeWolf",
    "Daniel Rasmussen",
    "Lisanne Huurdeman",
    "James Bergstra",
    "Eric Crawford",
    "Oliver Trujillo",
    "Connor Smith",
    "Carter Kolbeck",
    "Aziz Hurzook",
    "Yan Wu",
    "Bruce Bobier",
    "Eric Hochstein",
    "Marc Hurwitz",
    "Jonathan Gagne",
    "Charlie Tang",
    "Steven Leigh",
    "Albert Mallia",
    "Jacqueline Mok",
    "David MacNeil",
    "Jean-Frederic de Pasquale",
    "Tripp Bryan",
    "Elliott Lloyd",
    "Zhitnik Anatoly",
    "Brandon Aubie",
    "Abby Litt",
    "Oscar Tackstrom",
    "James Martens",
    "Jon Fishbein",
    "Avery Miller",
    "Mike Lerman",
    "Chris Parisien",
    "Ray Singh",
    "Dwight Kuo",
    "Phil Smith",
    "Harris Nover",
    "John Conklin",
    "Brian Fischer",
    "Brandon Westover",
    "Shahin Hakimian",
    "Qingfeng Huang",
    "Michael J. Barber",
    "Libby Buchholtz",
]


def analyse_file(fn):
    publications = json.load(open(fn, 'r'))
    publications = list(
        sorted(filter(lambda x: x["year"] and x["year"] > 2003 and not x["id"] in blacklist, publications),
               key=lambda x: x["year"]))

    def fuzzy_match_name(n1, n2):
        def canonicalise(s):
            return unidecode(s).lower()

        parts1 = list(map(canonicalise, filter(bool, re.split("[^\w]", n1))))
        parts2 = list(map(canonicalise, filter(bool, re.split("[^\w]", n2))))
        if len(parts1) == 0 or len(parts2) == 0:
            return False

        # Perfect match
        if parts1 == parts2:
            return True

        # Does the last name match?
        if parts1[-1] == parts2[-1]:
            # Does at least one of the first names match prefix-wise?
            for p1 in parts1[:-1]:
                for p2 in parts2[:-1]:
                    if p1.startswith(p2) or p2.startswith(p1):
                        return True

        # Are there at least two names that fully match?
        n_matches = 0
        for p1 in parts1:
            for p2 in parts2:
                if (len(p1) > 1) and (p1 == p2):
                    n_matches += 1
                    if n_matches >= 2:
                        return True

        return False

    def author_is_in_group(author, group):
        for group_member in group:
            if fuzzy_match_name(group_member, author):
                return True
        return False


#    for publication in publications:
#        print(publication["id"])
#        print("\t", publication["title"])
#        print("\t", publication["year"])
#        print("\t", publication["fieldsOfStudy"], publication["journalName"])
#        print()

# Sort by author

    cnrg_authors = set()
    non_cnrg_authors = set()
    for publication in publications:
        for author in publication["authors"]:
            name = author["name"]
            if (name in cnrg_authors) or author_is_in_group(
                    author["name"], cnrg_people):
                cnrg_authors.add(author["name"])
            else:
                non_cnrg_authors.add(author["name"])

    #print(cnrg_authors)
    #print(non_cnrg_authors)

    year_min = min(int(x["year"]) for x in publications)
    year_max = max(int(x["year"]) for x in publications)
    years = list(range(year_min, year_max + 1))

    n_papers_with_cnrg_first_author = {year: 0 for year in years}
    n_papers_with_cnrg_author = {year: 0 for year in years}
    n_papers = {year: 0 for year in years}

    for publication in publications:
        year = int(publication["year"])
        n_papers[year] += 1
        for i, author in enumerate(publication["authors"]):
            if author["name"] in cnrg_authors:
                if i == 0:
                    n_papers_with_cnrg_first_author[year] += 1
                n_papers_with_cnrg_author[year] += 1
                break

    # Sort by research topic
    n_neuromorphic = {year: 0 for year in years}
    n_biology = {year: 0 for year in years}
    n_theory = {year: 0 for year in years}
    n_software = {year: 0 for year in years}
    n_robotics = {year: 0 for year in years}

    # Initial semi-automatic categorisation
    patterns = {
        "other1":
        re.compile("(python)", re.IGNORECASE),
        "neuromorphics":
        re.compile(
            "(chip|neuromorphic|neural hardware|fpga|neurorobotic|neurosynaptic|digital hardware design|spinnaker|low-power|silicon|neural core)",
            re.IGNORECASE),
        "other2":
        re.compile("(nengo|nef approach|robotic)", re.IGNORECASE),
        "biology":
        re.compile(
            "(cerebell|hippocamp|serial memory|neurologically plausible|basal ganglia|biologically realistic|neural model|attention|smooth pursuit|path integration|speech production|speech processing|desert ant|hanoi|episodic memory|slow-wave sleep|working memory|neural memories|spatial reasoning|word associations|eye movement|auditory|gaze|spaun|fear-conditioning)",
            re.IGNORECASE),
    }
    for publication in publications:
        year = int(publication["year"])
        kind = kinds[publication["id"]] if (publication["id"]
                                            in kinds) else None
        for pattern_name, pattern in patterns.items():
            if not kind is None:
                break
            for field in ["title"]:
                if field in publication and pattern.search(publication[field]):
                    kind = pattern_name
                    break
        if kind == "neuromorphics":
            n_neuromorphic[year] += 1
        elif kind == "biology":
            n_biology[year] += 1
        elif kind == "theory":
            n_theory[year] += 1
        elif kind == "software":
            n_software[year] += 1
        elif kind == "robotics":
            n_robotics[year] += 1

        #color = ("92" if kind == "neuromorphics" else ("95" if kind == "biology" else "0"))
        #print(f"\033[{color}m", publication["id"], publication["title"], kind, "\033[0m")
        #sys.stdout.write(publication["id"] + "\t" + str(kind) + "\t" +
        #                 publication["title"] + "\n")

    y0 = np.array(list(n_papers_with_cnrg_first_author.values()))
    y1 = np.array(list(n_papers_with_cnrg_author.values()))
    y2 = np.array(list(n_papers.values()))

    z_neuromorphics = np.array(list(n_neuromorphic.values()))
    z_bio = np.array(list(n_biology.values()))
    z_theory = np.array(list(n_theory.values()))
    z_software = np.array(list(n_software.values()))
    z_robotics = np.array(list(n_robotics.values()))
    z_other = y2 - z_neuromorphics - z_bio - z_theory - z_software - z_robotics

    n0 = np.sum(y0)
    n1 = np.sum(y1)
    n2 = np.sum(y2)

    return {
        "y0": y0,
        "y1": y1,
        "y2": y2,
        "z_hw": z_neuromorphics,
        "z_bio": z_bio,
        "z_th": z_theory,
        "z_soft": z_software,
        "z_rob": z_robotics,
        "z_O": z_other,
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "year_min": year_min,
        "year_max": year_max,
        "years": years
    }

