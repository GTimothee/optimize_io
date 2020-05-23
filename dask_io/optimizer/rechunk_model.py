from math import floor

def model(m):
    print(f'\n--------Running new case--------')
    
    R, I, O, m = case["R"], case["I"], case["O"], case["m"]
    Lambd = I 
    B = [1, 1, Lambd[2]]
    n = 2

    for i in range(1):
        for j in range(1):
            for k in range(1):
                omega = [(i+1) * Lambd[0] % O[0], 
                         (j+1) * Lambd[1] % O[1],
                         (k+1) * Lambd[2] % O[2]]
                
                theta = [
                    Lambd[0] - omega[0],
                    Lambd[1] - omega[1],
                    Lambd[2] - omega[2]
                ]

                print(f'Omega: {omega}')
                print(f'Theta: {theta}')

                print(f'--------Storing F1--------')
                phi = floor(m / (omega[2] + Lambd[2]))
                print(f'Max value for Bj: {phi}')

                if phi >= theta[1]:
                    print(f'Setting Bj to thetaj')
                    B[1] = theta[1]
                else:
                    B[1] = phi if phi > 1 else 1
                    print("End of algorithm, Bj <- max_value")
                    return tuple(B)

                print(f'--------Storing F2 and F3--------')
                phi2 = floor( (m - theta[1] * (omega[2] + Lambd[2])) / ((n+1) * Lambd[2]))
                print(f'Max value for lambdaj - thetaj: {phi2}')
                print(f'Max value for Bj: {B[1] + phi2}')

                if (B[1] + phi2) >= Lambd[1]:   
                    print(f'Setting Bj to lambdaj')
                    B[1] = Lambd[1]
                else:
                    B[1] = B[1] + phi2 if phi2 > 1 else B[1]
                    print("End of algorithm, Bj <- max_value")
                    return tuple(B)

                print(f'--------Extending F1, F2 and F3--------')
                phi3 = floor( m / ( omega[2]*theta[1] + n*omega[1]*Lambd[2] + Lambd[1]*Lambd[2] ) )
                print(f'Max value for Bi: {phi3}')
                if phi3 >= theta[0]:
                    print(f'Setting Bi to thetai')
                    B[0] = theta[0]
                else:
                    B[0] = phi3 if phi3 > 1 else 1
                    print("End of algorithm, Bi <- max_value")
                    return tuple(B)

                return tuple(B)


if __name__ == "__main__":
    # WARNING: m is number of voxels, not bytes
    
    cases = [
        {   
            'R': (1,120,120),
            'I': (1,60,60),
            'O': (1,40,40),
            'm': 60*40 + (40*20), # buffer size + F1
            'Bexpected': (1,40,60)
        },{
            'R': (1,120,120),
            'I': (1,60,60),
            'O': (1,40,40),
            'm': 60*60 + 40*20 + 2*60*20, # buffer size + F1 + n(F2+F3)
            'Bexpected': (1,60,60)
        },{
            'R': (120,120,120),
            'I': (60,60,60),
            'O': (40,40,40),
            'm': 60*60*40 + 40*20*40 + 2*60*20*40, # buffer size + F1 + n(F2+F3)
            'Bexpected': (40,60,60)
        },
        {
            'R': (120,120,120),
            'I': (60,60,60),
            'O': (40,40,40),
            'm': 60*60 + 40*20 + 2*60*20, # buffer size + F1 + n(F2+F3)
            'Bexpected': (1,60,60)
        }
    ]

    for case in cases:
        B = model(case)
        print(f'Final buffer shape: {B}')
        try:
            assert case["Bexpected"] == B
            print("Success.")
        except:
            print('Bad output.')            