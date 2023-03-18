import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import PropTypes from 'prop-types';
import { makeStyles } from 'tss-react/mui';
import Typography from '@mui/material/Typography';
import Head from 'next/head';
import CssBaseline from '@mui/material/CssBaseline';
import { useTranslation } from 'next-i18next';
// Use this below for Server Side Render/Translation (SSR)
// import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
// Use this below for Static Site Generation (SSG)
import { getStaticPaths, makeStaticProps } from '/lib/getStatic';
import brand from '/public/text/brand';
import Header from '/components/Header';
// import Notification from '/components/Notification';

const useStyles = makeStyles({ uniqId: 'thanks' })(theme => ({
  mainWrap: {
    position: 'relative',
    width: '100%',
    overflow: 'hidden',
    background: theme.palette.mode === 'dark' ? theme.palette.background.default : theme.palette.background.paper,
  },
  spaceBottom: {
    marginBottom: theme.spacing(20)
  },
  spaceTop: {
    paddingTop: theme.spacing(1)
  },
  containerWrap: {
    marginTop: theme.spacing(1),
    padding: theme.spacing(4),
    '& > section': {
      position: 'relative'
    }
  },
}));

const loaderSupportersImage = ({ src, width, quality }) => {
  return `https://github.com/Fang-Zhang/Fang-Zhang.github.io/blob/main/public/images/supporters/${src}?w=${width}&q=${quality || 75}`
}

function ThanksPage(props) {
  const { classes, cx } = useStyles();
  const { onToggleDark, onToggleDir } = props;
  const { t } = useTranslation('common');

  return (
    <React.Fragment>
      <Head>
        <title>
          { brand.profile.name + ' - Thanks page' }
        </title>
      </Head>
      <CssBaseline />
      <div className={classes.mainWrap}>
        <Header
          onToggleDark={onToggleDark}
          onToggleDir={onToggleDir}
          invert
        />
        <main className={classes.containerWrap}>
          <section className={cx(classes.spaceTop, classes.spaceBottom)}>
            <Typography variant="h2" align="center" gutterBottom>
              {t('thanks_title')}
            </Typography>
            <Typography variant="h4" align="center">
              {t('thanks_subtitle')}  <Link href="https://github.com/sponsors/Fang-Zhang/" target="_blank">GitHub Sponsors</Link> and <Link href="https://patreon.com/user?u=63014909" target="_blank">Patreon</Link>.
            </Typography>
            <Image
              loader={loaderSupportersImage}
              src="MDearth.png"
              alt="Logo of the MDearth"
              width={266}
              height={112}
            />
            <br />
            <br />
            <Typography variant="h3" align="left">
              {t('thanks_premium_$1000_supporters')} 
            </Typography>
            <br />
            <br />
            <Typography variant="h3" align="left">
              {t('thanks_gold_$100_supporters')} 
            </Typography>
            <br />
            <br />
            <Typography variant="h3" align="left">
              {t('thanks_silver_$50_supporters')} 
            </Typography>
            <br />
            <br />
            <Typography variant="h3" align="left">
              {t('thanks_$10_supporters')} 
            </Typography>
            <br />
            <br />
          </section>
        </main>
        {/* <Notification /> */}
      </div>
    </React.Fragment>
  );
}

ThanksPage.propTypes = {
  onToggleDark: PropTypes.func.isRequired,
  onToggleDir: PropTypes.func.isRequired,
};

// Use this below for Server Side Render/Translation (SSR)
// export const getStaticProps = async ({ locale }) => ({ props: { ...await serverSideTranslations(locale, ['common']) } });

// Use this below for Static Site Generation (SSG)
const getStaticProps = makeStaticProps(['common']);
export { getStaticPaths, getStaticProps };

export default ThanksPage;
