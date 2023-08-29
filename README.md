# LSS_pix2pix
3D pix2pix GAN for matter density fields of the Large-Scale Structure

The aim of this model is to produce final matter density fields (z=0) from an input of initial fields (z=127). Each data sample has a size of 128 x 128 x 128. 
I have treated each sample as a grayscale data sample, put all the samples in a list and used this set to train a pix2pix model.
The success of the model will be measured by cosmological statistics such as the 2pcf. 

Another on-going experiment is to check if the same model can be used for the inverse of the problem, i.e., going from final fields to initial fields.

<!## Current Results

For a given input of an initial field (z=127), the visual comparison of the generated and true sample(z=0) is as shown:>



 
