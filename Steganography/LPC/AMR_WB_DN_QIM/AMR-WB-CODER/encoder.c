/*
 *===================================================================
 *  3GPP AMR Wideband Floating-point Speech Codec
 *===================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>
#include "typedef.h"
#include "enc_if.h"

#ifndef IF2
#define AMRWB_MAGIC_NUMBER "#!AMR-WB\n"
#endif


 

int main()
{
	int filenu, j;

	char Infile[100] = { 0 };
	char Outfile[100] = { 0 };
	double samtime[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	int em_rate[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	srand((unsigned)time(NULL));

	double real_rate;
	int frame = 150;
	for (int i = 0; i < 1; i++)//em_rate
	{
		for (int j = 0; j < 10; j++)//time
		{
			for (filenu = 1; filenu <= 20000; filenu++)
			{
				sprintf(Infile, "D:\\AudioData\\PCM\\English\\%.1fs\\English%05d.pcm", samtime[j], filenu);
				//sprintf(Infile, "D:\\DataSet\\trash\\WB\\Chinese%d.pcm", filenu);
				sprintf(Outfile, "D:\\AudioData\\AMR_WB_COVER\\English\\%.1fs\\English%05d.dat", samtime[j], filenu);
				printf("-------------------------------------------------------------------------\n");
				do {
					real_rate = 0;
					printf("%s\n%s\n%d\n", Infile, Outfile, filenu);
					int changeframe = 0;
					coder(Infile, Outfile, em_rate[i], &changeframe);
					real_rate = (double)(((double)changeframe / (double)(frame * 5)) * 100);
					printf("实际嵌入率：%.2f%%\n", real_rate);
				} while (real_rate != 0);//while (real_rate < (double)(em_rate[i] - 5) || real_rate >(double)(em_rate[i]));
				printf("-------------------------------------------------------------------------\n");
			}
		}
	}
	return 0;
}

/*
 * ENCODER.C
 *
 *    Usage : encoder (-dtx) mode speech_file  bitstream_file
 *
 *    Format for speech_file:
 *      Speech is read from a binary file of 16 bits data.
 *
 *    Format for bitstream_file:
 *        Described in TS26.201
 *
 *    mode = 0..8 (bit rate = 6.60 to 23.85 k)
 *
 *    -dtx if DTX is ON
 */
//int main(int argc, char *argv[])
//int coder (char *infilename, char *outfilename,int em_rate, int filenu){
int coder(char *infilename, char *outfilename, int em_rate, int *changeframe) {

	int argc;
	char **argv = (char**)malloc(4 * sizeof(char*));

	FILE *f_speech = NULL;                 /* File of speech data                   */
	FILE *f_serial = NULL;                 /* File of serial bits for transmission  */
	FILE *f_mode = NULL;                   /* File of modes for each frame          */

	Word32 serial_size, frame;
	Word16 signal[L_FRAME16k];             /* Buffer for speech @ 16kHz             */
	Word16 coding_mode = 0, allow_dtx, mode_file, mode = 8;
	UWord8 serial[NB_SERIAL_MAX];
	void *st;

	fprintf(stderr, "\n");
	fprintf(stderr, "===================================================================\n");
	fprintf(stderr, " 3GPP AMR-WB Floating-point Speech Coder, v7.0.0, Mar 20, 2007\n");
	fprintf(stderr, "===================================================================\n");
	fprintf(stderr, "\n");

	/*
	 * Open speech file and result file (output serial bit stream)
	 */


	argc = 4;
	argv[2] = infilename;
	argv[3] = outfilename;
	argv[1] = "8";

	if ((argc < 4) || (argc > 6))
	{
		fprintf(stderr, "Usage : encoder  (-dtx) mode speech_file  bitstream_file\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Format for speech_file:\n");
		fprintf(stderr, "  Speech is read form a binary file of 16 bits data.\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Format for bitstream_file:\n");
#ifdef IF2
		fprintf(stderr, "  Described in TS26.201.\n");
#else
		fprintf(stderr, "  Described in RFC 3267 (Sections 5.1 and 5.3).\n");
#endif
		fprintf(stderr, "\n");
		fprintf(stderr, "mode: 0 to 8 (9 bits rates) or\n");
		fprintf(stderr, "      -modefile filename\n");
		fprintf(stderr, " ===================================================================\n");
		fprintf(stderr, " mode   :  (0)  (1)   (2)   (3)   (4)   (5)   (6)   (7)   (8)     \n");
		fprintf(stderr, " bitrate: 6.60 8.85 12.65 14.25 15.85 18.25 19.85 23.05 23.85 kbit/s\n");
		fprintf(stderr, " ===================================================================\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "-dtx if DTX is ON, default is OFF\n");
		fprintf(stderr, "\n");
		exit(0);
	}
	allow_dtx = 0;




	if (strcmp(argv[1], "-dtx") == 0)
	{
		allow_dtx = 1;
		argv++;
	}
	mode_file = 0;
	if (strcmp(argv[1], "-modefile") == 0)
	{
		mode_file = 1;
		argv++;
		if ((f_mode = fopen(argv[1], "r")) == NULL)
		{
			fprintf(stderr, "Error opening input file  %s !!\n", argv[1]);
			exit(0);
		}
		fprintf(stderr, "Mode file:  %s\n", argv[1]);
	}
	else
	{
		mode = (Word16)atoi(argv[1]);
		if ((mode < 0) || (mode > 8))
		{
			fprintf(stderr, " error in bit rate mode %d: use 0 to 8\n", mode);
			exit(0);
		}
	}

	if ((f_speech = fopen(argv[2], "rb")) == NULL)
	{
		fprintf(stderr, "Error opening input file  %s !!\n", argv[2]);
		exit(0);
	}
	fprintf(stderr, "Input speech file:  %s\n", argv[2]);

	if ((f_serial = fopen(argv[3], "wb")) == NULL)
	{
		fprintf(stderr, "Error opening output bitstream file %s !!\n", argv[3]);
		exit(0);
	}
	fprintf(stderr, "Output bitstream file:  %s\n", argv[3]);

	/*
	 * Initialisation
	 */

	st = E_IF_init();

#ifndef IF2

	/* If MMS output is selected, write the magic number at the beginning of the
	 * bitstream file
	 */

	fwrite(AMRWB_MAGIC_NUMBER, sizeof(char), strlen(AMRWB_MAGIC_NUMBER), f_serial);

#endif

	/*
	 * Loop for every analysis/transmission frame.
	 *   -New L_FRAME data are read. (L_FRAME = number of speech data per frame)
	 *   -Conversion of the speech data from 16 bit integer to real
	 *   -Call coder to encode the speech.
	 *   -The compressed serial output stream is written to a file.
	 */

	fprintf(stderr, "\n --- Running ---\n");

	frame = 0;

	while (fread(signal, sizeof(Word16), L_FRAME16k, f_speech) == L_FRAME16k)
	{
		if (mode_file)
		{
			if (fscanf(f_mode, "%hd", &mode) == EOF)
			{
				mode = coding_mode;
				fprintf(stderr, "\n end of mode control file reached\n");
				fprintf(stderr, " From now on using mode: %hd.\n", mode);
				mode_file = 0;
			}

			if ((mode < 0) || (mode > 8))
			{
				fprintf(stderr, " error in bit rate mode %hd: use 0 to 8\n", mode);
				E_IF_exit(st);
				fclose(f_speech);
				fclose(f_serial);
				fclose(f_mode);
				exit(0);
			}
		}

		coding_mode = mode;

		frame++;
		/*fprintf(stderr, " Frames processed: %ld\r", frame);*/

		serial_size = E_IF_encode(st, coding_mode, signal, serial, allow_dtx, em_rate, changeframe);

		fwrite(serial, 1, serial_size, f_serial);

	}
	printf(" Frames processed: %d", frame);
	E_IF_exit(st);

	fclose(f_speech);
	fclose(f_serial);

	if (f_mode != NULL)
	{
		fclose(f_mode);
	}

	return 0;
}
